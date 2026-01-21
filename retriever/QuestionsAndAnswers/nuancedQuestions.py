from pathlib import Path
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.decomposition import TruncatedSVD
import yake
import time
from google.api_core.exceptions import ResourceExhausted
import glob
import re
import ast

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Define paths with the user-specified locations




class PaperEmbeddingAnalyzer:
    def __init__(self):
        # Initialize SciBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.keyword_extractor = yake.KeywordExtractor()
        self.topic_model = None
        self.fallback_mode = False
        

    def _initialize_topic_model(self, n_docs):
        """Initialize topic model based on dataset size"""
        if n_docs < 5:  # For very small datasets
            self.fallback_mode = True
            # Use SVD instead of UMAP for small datasets
            dim_reducer = TruncatedSVD(n_components=min(2, n_docs - 1))
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                prediction_data=True
            )
            
            self.topic_model = BERTopic(
                umap_model=dim_reducer,
                hdbscan_model=hdbscan_model,
                nr_topics=min(2, n_docs),
                verbose=True
            )
        else:
            self.fallback_mode = False
            umap_model = UMAP(
                n_neighbors=min(2, n_docs - 1),
                n_components=min(2, n_docs - 1),
                min_dist=0.0,
                metric='cosine'
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                prediction_data=True
            )
            
            self.topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                nr_topics="auto",
                verbose=True
            )
    
    def embed_text(self, text):
        """Generate embeddings for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def extract_keywords(self, text, top_k=5):
        """Extract key phrases from text using YAKE."""
        if not text or len(text.strip()) == 0:
            return ["no_keywords_found"]
        try:
            keywords = self.keyword_extractor.extract_keywords(text)
            keywords = sorted(keywords, key=lambda x: x[1])[:top_k]
            return [kw[0] for kw in keywords]
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return ["keyword_extraction_failed"]

    def analyze_paper(self, title, abstract, findings):
        """Generate a composite embedding by combining title, abstract, and findings embeddings."""
        try:
            title_emb = self.embed_text(title)
            abstract_emb = self.embed_text(abstract)
            findings_emb = self.embed_text(findings)
            
            # Composite embedding using weighted averages
            combined_embedding = torch.mean(torch.stack([title_emb * 1.5, abstract_emb, findings_emb]), dim=0)
            return combined_embedding
        
        except Exception as e:
            print(f"Error in analyze_paper: {e}")
            # Return a zero embedding as fallback
            return torch.zeros((1, 768))

    def fit_topic_model(self, documents):
        """Fit topic model on documents with fallback for small datasets"""
        try:
            if not documents or len(documents) == 0:
                print("No documents provided for topic modeling.")
                return

            print(f"Fitting topic model on {len(documents)} entries from papers")
            
            # Initialize appropriate model based on dataset size
            self._initialize_topic_model(len(documents))
            
            # For very small datasets, use simple topic assignment
            if self.fallback_mode:
                print("Using fallback mode for small dataset")
                self.topic_model.fit_transform(documents)
                print("Topic modeling completed in fallback mode")
            else:
                self.topic_model.fit_transform(documents)
                print("Topic modeling completed successfully")
                
        except Exception as e:
            print(f"Error during topic modeling: {e}")
            self.fallback_mode = True
            print("Falling back to simple topic assignment")
            # Create a simple fallback topic assignment
            self.topic_model = None

    def get_topics_for_paper(self, text):
        """Get topics for a single paper with fallback handling"""
        try:
            if self.fallback_mode or self.topic_model is None:
                return [("General Topic", 1.0)]
                
            topics, _ = self.topic_model.transform([text])
            return self.topic_model.get_topic(topics[0]) if topics[0] != -1 else [("General Topic", 1.0)]
        except Exception as e:
            print(f"Error getting topics: {e}")
            return [("General Topic", 1.0)]

#NOTA: el JSONL se genera con el nombre question results###

class NuancedQuestions:
    def __init__(self, embedding_analyzer, shared_timestamp=None):
        # Configure the Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.embedding_analyzer = embedding_analyzer
        self.PROJECT_DIR = Path(".")
        
        # Create reports/questions directory if it doesn't exist
        questions_dir = Path("./reports/questions")
        questions_dir.mkdir(parents=True, exist_ok=True)
        
        # Use shared timestamp if provided, otherwise generate new one
        timestamp = shared_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file = questions_dir / f"question_results_{timestamp}.jsonl"

        self.query_results_path = ""

        # Directory where your JSON files are stored
        directory = '.'  # Replace with your actual path


        # Pattern to match filenames and extract timestamp
        pattern = re.compile(r"combined_report_(\d{8}_\d{6})\.json")

        # Find all JSON files matching the pattern - check reports/combined first
        json_files = glob.glob("./reports/combined/combined_report_*.jsonl")
        if not json_files:
            # Fallback to root directory for backward compatibility
            json_files = glob.glob(os.path.join(directory, "combined_report_*.jsonl"))

        # Extract timestamps and find the most recent file
        most_recent_file = None
        latest_timestamp = None

        for file in json_files:
            match = pattern.search(os.path.basename(file))
            if match:
                timestamp = match.group(1)
                # Convert timestamp to an integer for comparison
                timestamp_int = int(timestamp)
                # Update the most recent file if the timestamp is newer
                if latest_timestamp is None or timestamp_int > latest_timestamp:
                    latest_timestamp = timestamp_int
                    most_recent_file = file

        # Store the most recent path in query_results_path

        self.query_results_path = most_recent_file
        print("this queryyyy", self.query_results_path)

    def load_relevant_papers(self, filename):
        """Load query results from a JSONL file and extract unique sources."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                # Read the JSONL file line by line
                query_results = [json.loads(line) for line in file]
            
            print(f"Successfully loaded query results from {filename}")
            
            # Extract unique sources
            unique_sources = {entry["Source"] for entry in query_results}
            print("unique sources: ", unique_sources)
            print(f"Number of relevant sources found: {len(unique_sources)}")
            
            return unique_sources
        
        except Exception as e:
            print(f"Error loading query results: {e}")
            return None

    def extract_text_from_pdf(self, paper_id):
        # Use paper_id directly - it's already a full path
        pdf_path = paper_id
        
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
            return text
        except FileNotFoundError:
            print(f"File {pdf_path} not found.")
            return ""

    def extract_sections(self, paper_text):
        """Extracts title, abstract, and findings sections from paper text."""
        title = paper_text.split("\n")[0]  # Assume the title is the first line
        abstract = paper_text[:len(paper_text) // 3]
        findings = paper_text[2 * len(paper_text) // 3:]
        return title, abstract, findings

    def generate_questions(self, topic, keywords, paper_embedding):
        """Generate three questions for each paper based on its main topic, keywords, and embedding."""
        prompt = (
            f"Based on the following topic and keywords, generate three questions that "
            f"help capture the nuances and specific approach of the corresponding paper"
            f"Topic: {topic}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Format the output as a python list of strings with double quotes format and looks like this [question 1, question 2, ...] in which each element only contains the question, no enumeration. Make sure that the output is only a string that looks like a python list"
        )
        
        response = self._call_with_retry(lambda: self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "max_output_tokens": 300}
        ))

        if response and response.text.strip():
            response_text= response.text
            # Extract the Python code block content
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            response_list = response_text[start_idx:end_idx]

            # Convert the string representation of the list into an actual Python list
            response_clean = ast.literal_eval(response_list)
            return response_clean
        else:
            return ["Error generating questions."]

    def save_questions(self, paper_id, questions):
        """Save questions in a JSONL file with paper ID and generated questions."""
        with open(self.output_file, "a", encoding="utf-8") as f:
            json.dump({"paper_id": paper_id, "questions": questions}, f)
            f.write("\n")

    def analyze_and_generate_questions(self, external_contents, external_content_by_title):
        """Process each relevant paper to extract topics, keywords, and generate questions."""
        relevant_papers_ids = self.load_relevant_papers(self.query_results_path)
        if not relevant_papers_ids:
            return

        # Collect all paper texts for topic modeling
        all_texts = []
        
        for paper_id in relevant_papers_ids:
            if paper_id not in external_content_by_title.keys():
                paper_text = self.extract_text_from_pdf(paper_id)
                if paper_text:
                    all_texts.append(paper_text)

        all_texts = all_texts + external_contents
        # Fit BERTopic on all documents
        self.embedding_analyzer.fit_topic_model(all_texts)

        # Process each paper to generate questions
        for paper_id in relevant_papers_ids:
            if paper_id in external_content_by_title.keys():
                paper_text = external_content_by_title[paper_id]
            else:
                paper_text = self.extract_text_from_pdf(paper_id)
            
            if paper_text:
                # Extract sections and generate embeddings
                title, abstract, findings = self.extract_sections(paper_text)
                keywords = self.embedding_analyzer.extract_keywords(paper_text)
                paper_embedding = self.embedding_analyzer.analyze_paper(title, abstract, findings)
                
                # Extract topic for this paper
                topics = self.embedding_analyzer.get_topics_for_paper(paper_text)
                main_topic = topics[0][0] if topics else "No main topic found"
                
                # Generate three comparison-focused questions per paper
                questions = self.generate_questions(main_topic, keywords, paper_embedding)
                
                # Save questions to JSONL
                self.save_questions(paper_id, questions)

    def _call_with_retry(self, func, retries=3, backoff=2):
        """Helper method to handle retries with exponential backoff on API limit errors."""
        for attempt in range(retries):
            try:
                return func()
            except ResourceExhausted:
                wait_time = backoff ** attempt
                print(f"API limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        print("API limit exceeded and retry attempts exhausted.")
        return None

    def run(self, external_contents, external_content_by_title):
        """Run the analyzer to generate questions."""

        self.analyze_and_generate_questions(external_contents= external_contents, external_content_by_title=external_content_by_title)


if __name__ == "__main__":
    print("Generating nuanced questions.... ")
    embedding_analyzer = PaperEmbeddingAnalyzer()
    analyzer = NuancedQuestions(embedding_analyzer)
    all_external_content = ['Assessing the needs of women who use drugs requires a comprehensive understanding of gender and intersectionality. Gender refers to socially constructed roles that vary based on time and place, and gender identity reflects one\'s internal sense of being a woman, man, or anywhere along the gender spectrum, including transgender, nonbinary, and genderqueer identities. In this article and in the fellowship track, we define \'women\' as all individuals who identify as a woman, regardless of their sex (classification as male or female based on biological attributes). Intersectional perspectives recognize that women\'s experiences with drug use are not homogeneous. Rather, other intersecting identities, such as gender identity, sexual orientation, race/ethnicity, and socioeconomic class shape individual experiences of oppression or empowerment [5]. In particular, structural racism, homophobia, and transphobia enhance discrimination and treatment barriers for Black, Indigenous, and other racialized individuals and for transgender and genderqueer individuals compared to White cis-gender women who use drugs.\n\nWomen who use drugs interact with individuals, communities, and social systems that reproduce structural sexism. Structural sexism is defined as "discriminatory beliefs or practices on the basis of sex and gender that are entrenched in societal frameworks and which result in fairly predictable disparities in social outcomes related to power, resources, and opportunities" [7]. For example, gender-based power dynamics in drug-using communities may restrict women\'s autonomy to determine when, how, and why they use drugs. Such power imbalances are associated with greater adverse consequences in women compared to men including higher rates of injection drug use-associated infections, co-occurring mood and anxiety disorders, and experiences of intimate partner violence and sexual exploitation [8,9].\n\nStructural sexism is also apparent in the systems that affect pregnant and parenting people who use drugs. Pregnant individuals who use drugs face punitive consequences from legal and child welfare systems, hostility from the general public, and an addiction treatment system that is poorly suited to meet their needs. The child welfare system has traditionally viewed prenatal and parental substance use as synonymous with abuse or neglect, causing heightened shame, stigma, and fear of seeking treatment. Black and Indigenous women are disproportionately harmed by trauma related to child welfare service reporting and custody loss.', "This research contributes to a growing body of literature on integrated harm reduction approaches for supporting women, girls, and gender diverse people with substance use challenges and other complex needs.In particular, this study informs our understanding and subsequent ability to address the needs of women and girls who access substance use treatment, with consideration of factors that may support program completion and long-term success.Moreover, this study has the potential to guide evolving best practice at the 2nd Floor Women's Recovery Centre and beyond and has significant policy implications with respect to the prioritization, design, and implementation of interventions and frameworks that reduce the likelihood of substance-exposed pregnancies, ultimately supporting health and wellbeing for parents, children, families, and communities.", '1. Understanding sex, gender, and gender differences in drug use 2. Structural sexism and intersectionality 3. Substance use in adolescents and young adults, including girls under age 18 4. Substance use in transgender, non-binary, and genderqueer populations 5. Perinatal treatment of SUD, including management of OUD 6. Neonatal withdrawal syndromes 7. Child welfare system involvement in people with SUD 8. Contraception and abortion for people who use drugs 9. Gender-responsive care 10. Sex work and substance use 11. Intimate partner violence and substance use 12. Trauma-informed care for women who use drugs 13. Co-occurring SUD and psychiatric disorders Self-directed learning resources: Educational activities and practice resources:  \n\n1. Understanding sex, gender, and gender differences in drug use 2. Structural sexism and intersectionality 3. Substance use in adolescents and young adults, including girls under age 18 4. Substance use in transgender, non-binary, and genderqueer populations 5. Perinatal treatment of SUD, including management of OUD 6. Neonatal withdrawal syndromes 7. Child welfare system involvement in people with SUD 8. Contraception and abortion for people who use drugs 9. Gender-responsive care 10. Sex work and substance use 11. Intimate partner violence and substance use 12. Trauma-informed care for women who use drugs 13. Co-occurring SUD and psychiatric disorders Self-directed learning resources: Educational activities and practice resources:', 'Our findings suggest that addressing the basics needs such as food, clothing, safety, and housing of female drug users who lack these necessities may support improved access to care. State Medicaid programs may be well-positioned to lead on this front given their flexibility in program design, waiver authority, ability to offer provider incentive payments, and the patient populations they support. 25 In California, one novel approach to integrating health and social needs is the Whole Person Care Medicaid Section 1115 wavier demonstration, launched in 2016, which provides patients from stateidentified high risk groups (including patients with substance use disorder) with integrated local systems of healthcare, behavioral, and social services. 26 Implemented at the local level, the Whole Person Care pilot programs coordinate services through partnerships between health agencies, the social safety net, and Medicaid managed care plans.\n\nHealthcare providers should be emboldened with the time and resources to adequately address the comorbidity of substance abuse and mental illness. This would allow for better detection of substance abuse and mental illness comorbidity to be identified and treated regardless of the financial resources of the patient. Furthermore, health systems should create safe spaces that allow women with limited social stability to be able to access care free of charge to remove the burden of cost from care. 27 To contend with drug dependence, we must address mental health issues and create curated programs that address the specific needs of women who use drugs, focusing on the demonstrated effectiveness of targeted interventions rather than criminalization. [28][29][30][31] Moving forward, studies should focus on drug use among specific ethnic minority populations. There is a lack of literature on the impact of social determinants of health on mental health access of the Black and Hispanic population who use illicit drugs, as well as studies that directly compare drug user and non-user health outcomes. Future research should investigate the effectiveness of gender-based drug program treatment with a special focus on the needs of female drug users with mental health disorders.', 'Little research to date has examined interventions designed specifically to address substance use problems in women on welfare. Generally, studies indicate that these women have myriad co-occurring problems in the areas of mental health, domestic violence, and medical care, as well as legal issues. Thus, interventions designed to specifically address substance abuse may not effectively address the significant and chronic problems experienced by women receiving welfare benefits. Interventions that provide gender-specific services and coordination across multiple service domains to address the co-occurring problems these women experience may be most effective.\n\nTwo recent studies examined the effectiveness of case management (CM) at addressing the multiple problems experienced by substance-abusing women on welfare: CASAWORKS for Families (CWF) (Morgenstern et al. 2003b) and CASASARD (Morgenstern et al. 2001b). These CM interventions, designed specifically for TANF women, provided linkages to needed wraparound services in many areas, including housing assistance, mental health treatment, medical treatment, child care, and transportation. Additionally, when possible, services were tailored to women by referring clients to treatment programs that had female therapists, women-only groups, and child care.\n\nCWF was a demonstration program testing an inten sive intervention for TANF women with substance use problems in 10 counties around the country (Morgenstern et al. 2003b). CWF offered client-level case management and fostered interagency coordination to ensure that clients had access to ancillary services. The study did not employ a control group, but researchers conducted a rigorous evaluation of CWF with 698 women receiving treatment at 10 sites. An independent evaluation of this demonstration project produced promising findings (McLellan et al. 2003). Women had high rates of reten tion (51 percent were still in treatment 6 months after beginning treatment) and received substantial amounts of ancillary services. Followup at 12 months showed Prevalence of barriers to employment among substance-abusing and non-substance-abusing female welfare recipients. On average, more than twice as many substance-abusing women experienced severe barriers to employment com pared with non-substance-abusing women. that the women had significant and meaningful reductions in substance use (78 percent reported no heavy alcohol use in the previous 6 months), increases in employment (41 percent were employed at least part-time at the 12month followup), and decreases in welfare dependency.', 'The human and economic costs of substance use are considerable [1,2]. Although rates of substance use generally are lower for women than for men [3][4][5], the physical and mental health consequences can be more profound for women [6]. Women who use alcohol and illicit drugs are at particular risk for hepatitis C and HIV infection, and are more likely to have psychiatric co-morbidity and multimorbidity [7]. In addition, substance use during pregnancy and while mothering has negative consequences for children, including risk for prematurity, impaired physical growth and development, physical and mental health problems, and development of substance use problems [8][9][10][11]. There is a need for services that effectively and comprehensively address the complex needs of women with substance use issues and their children. In addition to experiencing physical and mental health problems, these women often have personal histories of exposure to physical and sexual abuse and other relationship problems, negative or inadequate social support systems, inadequate income, unemployment, unstable housing, and involvement with the criminal justice system [12][13][14]. Conners and colleagues [9] suggested that an accumulation of these postnatal environmental risk conditions combined with prenatal substance exposure results in increased childhood vulnerability to poor outcomes. As these authors note, the issues mothers face can "limit their ability to provide for their child\'s physical and/or emotional needs" (p. 90). Maternal substance use has been associated with limited parenting capacity and an increased likelihood that children are exposed to maltreatment, including neglect [8,[15][16][17], factors that have negative developmental sequelae for children. Children of women with substance use issues are further compromised because they have limited opportunities to develop the social skills and relationships that can help to buffer against risk [9].', "levels of homelessness (58%) and food insecurity (89.5%).Conclusion: Study findings underscore the need for better understanding of the existing capabilities of WESW and those who use drugs, including financial autonomy and communityThis is an open-access article distributed under the terms and conditions of the Creative Commons Attribution license supports, that may guide the design of programs that most effectively promote women's economic well-being and ensure that it is not at the expense of wellness and safety. Designing such programs requires incorporating a social justice lens into social work and public health interventions, including HIV prevention, and attention to the human rights of the most marginalized and highest risk populations, including WESW and those who use drugs.Keywordswomen engaged in sex work; FSW; drug use; financial lives; paradoxical autonomy Yang et al.", 'Women comprise one-third of people who use drugs globally and account for one-fifth of the estimated global number of people who inject drugs [1].Women comprised one-third of overdose deaths in the US and about one in four in Canada in 2017-18.The rate of fatal overdoses among women has increased by 260-500% in the last two decades [2].Women also suffer serious longterm social and health consequences of incarceration related to drug use and drug-related offenses which are different to those suffered by men [3][4][5].The latest European report on Women and Drugs estimated one in four people with serious drug problems and one in five entrants to treatment programs were women.Despite the disease burden, the report lamented limited availability of integrated and coordinated national-level genderspecific services and gender-mainstreaming responses of drug use-related problems [6]. Women face particular challenges related to drugs including gender, effects of drug use during pregnancy (e.g., neonatal abstinence syndrome, low birth weight, and premature birth), motherhood, gender-based violence, higher involvement in sex work, higher prevalence of (sexual) trauma, double stigma (being discriminated against for being a woman and persons who use drugs) with serious psychosocial consequences [7].These challenges require gender-specific policy responses.Drug policy should be well-aligned with the objectives of sustainable development goals (SDG-2030), which envisage gender equality and empowerment [8].Therefore, it is important to assess whether women who use drugs currently receive attention in drug policies and programs and in what ways.Assessing gender-specific elements of national', "assets [18]. If such guidelines are not followed, they may lose future funding and are penalized for every dollar they go over the designated maximum alliance for individual assets. Even if there may be a desire to work and to save money, government regulations may produce a disincentive to both. For women who are recovering from substance dependence and other related adverse life events (such as HIV infection, poor health, intimate partner violence, and mental health concerns), it may be difficult to see the value of engaging in economic empowerment activities with so little economic independence. This paper attempts to explore these issues to enhance our knowledge of effective interventions that support women on disability living in low-income communities and high-risk environments to gain access to the workforce. The Women's Economic Empowerment pilot (WEE) described here was initiated to better understand the acceptability and feasibility of implementing a structural intervention to promote women's economic empowerment in an urban context. Grounded in social cognitive [19,20] and asset theories [21], the intervention targeted behavior change by building women's self-efficacy in sexual health decision making and condom negotiation and use, while also promoting economic stability through training in financial literacy and the accumulation of economic assets. The study aimed to (1) explore whether a structural intervention combining health promotion with economic empowerment activities would be both acceptable and feasible among women receiving HIV prevention and related case management services in NYC and (2) obtain preliminary data that would support the design of a future efficacy trial. If we better understand the successes and challenges of implementation, we may improve the overall effectiveness of combination, structural economic empowerment interventions for low-income communities. As such, Figure 1 depicts the conceptual model for the combined intervention.", "Sample characteristics, stratified by gender identity, are reported in Table 2. Participants were on average 41 years old (SD = 12); and were mostly of Malay (42%), Indian (42%), or Chinese (9%) ethnicity. The majority of participants identified as Muslim (55%), followed by Hindu (27%), Buddhist (9%), Christian (6%), or Sikh (3%). Many participants reported not completing secondary school (45%) and less than half (46%) had children. Participants' demographic traits differed according to their gender identity, with a substantially higher proportion of TWSWs reporting being single (87%), having no children (87%), and holding some form of secondary schooling qualification (60%) when compared to CWSW. Moreover, a larger percentage of CWSWs reported being widowed/divorced (72%) in comparison to their TWSW counterparts.  Table 3 reports participants' patterns of income generation, stratified by gender identity. Participants reported having engaged in sex work for an average of 20 years (SD = 12); although, both CWSWs (61%) and TWSWs (72%) reported receiving income from non-sex work forms of work, including cleaning services, entertainment, and night club promotion. Participants' total mean monthly income was MYR 2235 (USD 521). Table 4 shows rates of lifetime drug use, stratified by gender identity. The most commonly used drugs were amphetaminetype substances (ATS) (50%), primarily crystal methamphetamine, followed by cannabis (36%) and heroin (27%). Engagement in drug use differed according to respondents' gender identity; with a higher percentage of CWSWs using ATS (61%) and/or heroine (44%) in comparison to TWSWs.\n\nThe in-depth interviews yielded three main themes regarding acceptability of a microfinance intervention: (a) participants were eager to engage in additional forms of income generation due to familial concerns and career aspirations; (b) interest in all proposed components of the intervention were driven by a desire to build their own business; and (c) potential challenges to developing businesses included a lack of financial resources, competition from other businesses, and fear of stigma."]
    content_by_title = {
        "Developing A Women'S Health Track Within Addiction Medicine Fellowship: Reflections And Inspirations": 'Assessing the needs of women who use drugs requires a comprehensive understanding of gender and intersectionality. Gender refers to socially constructed roles that vary based on time and place, and gender identity reflects one\'s internal sense of being a woman, man, or anywhere along the gender spectrum, including transgender, nonbinary, and genderqueer identities. In this article and in the fellowship track, we define \'women\' as all individuals who identify as a woman, regardless of their sex (classification as male or female based on biological attributes). Intersectional perspectives recognize that women\'s experiences with drug use are not homogeneous. Rather, other intersecting identities, such as gender identity, sexual orientation, race/ethnicity, and socioeconomic class shape individual experiences of oppression or empowerment [5]. In particular, structural racism, homophobia, and transphobia enhance discrimination and treatment barriers for Black, Indigenous, and other racialized individuals and for transgender and genderqueer individuals compared to White cis-gender women who use drugs.\n\nWomen who use drugs interact with individuals, communities, and social systems that reproduce structural sexism. Structural sexism is defined as "discriminatory beliefs or practices on the basis of sex and gender that are entrenched in societal frameworks and which result in fairly predictable disparities in social outcomes related to power, resources, and opportunities" [7]. For example, gender-based power dynamics in drug-using communities may restrict women\'s autonomy to determine when, how, and why they use drugs. Such power imbalances are associated with greater adverse consequences in women compared to men including higher rates of injection drug use-associated infections, co-occurring mood and anxiety disorders, and experiences of intimate partner violence and sexual exploitation [8,9].\n\nStructural sexism is also apparent in the systems that affect pregnant and parenting people who use drugs. Pregnant individuals who use drugs face punitive consequences from legal and child welfare systems, hostility from the general public, and an addiction treatment system that is poorly suited to meet their needs. The child welfare system has traditionally viewed prenatal and parental substance use as synonymous with abuse or neglect, causing heightened shame, stigma, and fear of seeking treatment. Black and Indigenous women are disproportionately harmed by trauma related to child welfare service reporting and custody loss. 1. Understanding sex, gender, and gender differences in drug use 2. Structural sexism and intersectionality 3. Substance use in adolescents and young adults, including girls under age 18 4. Substance use in transgender, non-binary, and genderqueer populations 5. Perinatal treatment of SUD, including management of OUD 6. Neonatal withdrawal syndromes 7. Child welfare system involvement in people with SUD 8. Contraception and abortion for people who use drugs 9. Gender-responsive care 10. Sex work and substance use 11. Intimate partner violence and substance use 12. Trauma-informed care for women who use drugs 13. Co-occurring SUD and psychiatric disorders Self-directed learning resources: Educational activities and practice resources:  \n\n1. Understanding sex, gender, and gender differences in drug use 2. Structural sexism and intersectionality 3. Substance use in adolescents and young adults, including girls under age 18 4. Substance use in transgender, non-binary, and genderqueer populations 5. Perinatal treatment of SUD, including management of OUD 6. Neonatal withdrawal syndromes 7. Child welfare system involvement in people with SUD 8. Contraception and abortion for people who use drugs 9. Gender-responsive care 10. Sex work and substance use 11. Intimate partner violence and substance use 12. Trauma-informed care for women who use drugs 13. Co-occurring SUD and psychiatric disorders Self-directed learning resources: Educational activities and practice resources:',
        'Integrated Supports For Women And Girls Experiencing Substance Use And Complex Needs': "This research contributes to a growing body of literature on integrated harm reduction approaches for supporting women, girls, and gender diverse people with substance use challenges and other complex needs.In particular, this study informs our understanding and subsequent ability to address the needs of women and girls who access substance use treatment, with consideration of factors that may support program completion and long-term success.Moreover, this study has the potential to guide evolving best practice at the 2nd Floor Women's Recovery Centre and beyond and has significant policy implications with respect to the prioritization, design, and implementation of interventions and frameworks that reduce the likelihood of substance-exposed pregnancies, ultimately supporting health and wellbeing for parents, children, families, and communities.",
        'Social Stability And Unmet Health Care Needs In A Community-Based Sample Of Women Who Use Drugs': 'Our findings suggest that addressing the basics needs such as food, clothing, safety, and housing of female drug users who lack these necessities may support improved access to care. State Medicaid programs may be well-positioned to lead on this front given their flexibility in program design, waiver authority, ability to offer provider incentive payments, and the patient populations they support. 25 In California, one novel approach to integrating health and social needs is the Whole Person Care Medicaid Section 1115 wavier demonstration, launched in 2016, which provides patients from stateidentified high risk groups (including patients with substance use disorder) with integrated local systems of healthcare, behavioral, and social services. 26 Implemented at the local level, the Whole Person Care pilot programs coordinate services through partnerships between health agencies, the social safety net, and Medicaid managed care plans.\n\nHealthcare providers should be emboldened with the time and resources to adequately address the comorbidity of substance abuse and mental illness. This would allow for better detection of substance abuse and mental illness comorbidity to be identified and treated regardless of the financial resources of the patient. Furthermore, health systems should create safe spaces that allow women with limited social stability to be able to access care free of charge to remove the burden of cost from care. 27 To contend with drug dependence, we must address mental health issues and create curated programs that address the specific needs of women who use drugs, focusing on the demonstrated effectiveness of targeted interventions rather than criminalization. [28][29][30][31] Moving forward, studies should focus on drug use among specific ethnic minority populations. There is a lack of literature on the impact of social determinants of health on mental health access of the Black and Hispanic population who use illicit drugs, as well as studies that directly compare drug user and non-user health outcomes. Future research should investigate the effectiveness of gender-based drug program treatment with a special focus on the needs of female drug users with mental health disorders.',
        'Welfare Reform And Substance Abuse Treatment For Welfare Recipients': 'Little research to date has examined interventions designed specifically to address substance use problems in women on welfare. Generally, studies indicate that these women have myriad co-occurring problems in the areas of mental health, domestic violence, and medical care, as well as legal issues. Thus, interventions designed to specifically address substance abuse may not effectively address the significant and chronic problems experienced by women receiving welfare benefits. Interventions that provide gender-specific services and coordination across multiple service domains to address the co-occurring problems these women experience may be most effective.\n\nTwo recent studies examined the effectiveness of case management (CM) at addressing the multiple problems experienced by substance-abusing women on welfare: CASAWORKS for Families (CWF) (Morgenstern et al. 2003b) and CASASARD (Morgenstern et al. 2001b). These CM interventions, designed specifically for TANF women, provided linkages to needed wraparound services in many areas, including housing assistance, mental health treatment, medical treatment, child care, and transportation. Additionally, when possible, services were tailored to women by referring clients to treatment programs that had female therapists, women-only groups, and child care.\n\nCWF was a demonstration program testing an inten sive intervention for TANF women with substance use problems in 10 counties around the country (Morgenstern et al. 2003b). CWF offered client-level case management and fostered interagency coordination to ensure that clients had access to ancillary services. The study did not employ a control group, but researchers conducted a rigorous evaluation of CWF with 698 women receiving treatment at 10 sites. An independent evaluation of this demonstration project produced promising findings (McLellan et al. 2003). Women had high rates of reten tion (51 percent were still in treatment 6 months after beginning treatment) and received substantial amounts of ancillary services. Followup at 12 months showed Prevalence of barriers to employment among substance-abusing and non-substance-abusing female welfare recipients. On average, more than twice as many substance-abusing women experienced severe barriers to employment com pared with non-substance-abusing women. that the women had significant and meaningful reductions in substance use (78 percent reported no heavy alcohol use in the previous 6 months), increases in employment (41 percent were employed at least part-time at the 12month followup), and decreases in welfare dependency.',
        'Harm Reduction Journal Integrated Programs For Women With Substance Use Issues And Their Children: A Qualitative Meta-Synthesis Of Processes And Outcomes': 'The human and economic costs of substance use are considerable [1,2]. Although rates of substance use generally are lower for women than for men [3][4][5], the physical and mental health consequences can be more profound for women [6]. Women who use alcohol and illicit drugs are at particular risk for hepatitis C and HIV infection, and are more likely to have psychiatric co-morbidity and multimorbidity [7]. In addition, substance use during pregnancy and while mothering has negative consequences for children, including risk for prematurity, impaired physical growth and development, physical and mental health problems, and development of substance use problems [8][9][10][11]. There is a need for services that effectively and comprehensively address the complex needs of women with substance use issues and their children. In addition to experiencing physical and mental health problems, these women often have personal histories of exposure to physical and sexual abuse and other relationship problems, negative or inadequate social support systems, inadequate income, unemployment, unstable housing, and involvement with the criminal justice system [12][13][14]. Conners and colleagues [9] suggested that an accumulation of these postnatal environmental risk conditions combined with prenatal substance exposure results in increased childhood vulnerability to poor outcomes. As these authors note, the issues mothers face can "limit their ability to provide for their child\'s physical and/or emotional needs" (p. 90). Maternal substance use has been associated with limited parenting capacity and an increased likelihood that children are exposed to maltreatment, including neglect [8,[15][16][17], factors that have negative developmental sequelae for children. Children of women with substance use issues are further compromised because they have limited opportunities to develop the social skills and relationships that can help to buffer against risk [9].',
        'The Financial Lives And Capabilities Of Women Engaged In Sex Work: Can Paradoxical Autonomy Inform Intervention Strategies? Hhs Public Access': "levels of homelessness (58%) and food insecurity (89.5%).Conclusion: Study findings underscore the need for better understanding of the existing capabilities of WESW and those who use drugs, including financial autonomy and communityThis is an open-access article distributed under the terms and conditions of the Creative Commons Attribution license supports, that may guide the design of programs that most effectively promote women's economic well-being and ensure that it is not at the expense of wellness and safety. Designing such programs requires incorporating a social justice lens into social work and public health interventions, including HIV prevention, and attention to the human rights of the most marginalized and highest risk populations, including WESW and those who use drugs.Keywordswomen engaged in sex work; FSW; drug use; financial lives; paradoxical autonomy Yang et al.",
        "Drug Policies' Sensitivity Towards Women, Pregnancy, And Motherhood: A Content Analysis Of National Policy And Programs From Nine Countries And Their Adherence To International Guidelines": 'Women comprise one-third of people who use drugs globally and account for one-fifth of the estimated global number of people who inject drugs [1].Women comprised one-third of overdose deaths in the US and about one in four in Canada in 2017-18.The rate of fatal overdoses among women has increased by 260-500% in the last two decades [2].Women also suffer serious longterm social and health consequences of incarceration related to drug use and drug-related offenses which are different to those suffered by men [3][4][5].The latest European report on Women and Drugs estimated one in four people with serious drug problems and one in five entrants to treatment programs were women.Despite the disease burden, the report lamented limited availability of integrated and coordinated national-level genderspecific services and gender-mainstreaming responses of drug use-related problems [6]. Women face particular challenges related to drugs including gender, effects of drug use during pregnancy (e.g., neonatal abstinence syndrome, low birth weight, and premature birth), motherhood, gender-based violence, higher involvement in sex work, higher prevalence of (sexual) trauma, double stigma (being discriminated against for being a woman and persons who use drugs) with serious psychosocial consequences [7].These challenges require gender-specific policy responses.Drug policy should be well-aligned with the objectives of sustainable development goals (SDG-2030), which envisage gender equality and empowerment [8].Therefore, it is important to assess whether women who use drugs currently receive attention in drug policies and programs and in what ways.Assessing gender-specific elements of national',
        'Demonstrating The Feasibility Of An Economic Empowerment And Health Promotion Intervention Among Low-Income Women Affected By Hiv In New York City': "assets [18]. If such guidelines are not followed, they may lose future funding and are penalized for every dollar they go over the designated maximum alliance for individual assets. Even if there may be a desire to work and to save money, government regulations may produce a disincentive to both. For women who are recovering from substance dependence and other related adverse life events (such as HIV infection, poor health, intimate partner violence, and mental health concerns), it may be difficult to see the value of engaging in economic empowerment activities with so little economic independence. This paper attempts to explore these issues to enhance our knowledge of effective interventions that support women on disability living in low-income communities and high-risk environments to gain access to the workforce. The Women's Economic Empowerment pilot (WEE) described here was initiated to better understand the acceptability and feasibility of implementing a structural intervention to promote women's economic empowerment in an urban context. Grounded in social cognitive [19,20] and asset theories [21], the intervention targeted behavior change by building women's self-efficacy in sexual health decision making and condom negotiation and use, while also promoting economic stability through training in financial literacy and the accumulation of economic assets. The study aimed to (1) explore whether a structural intervention combining health promotion with economic empowerment activities would be both acceptable and feasible among women receiving HIV prevention and related case management services in NYC and (2) obtain preliminary data that would support the design of a future efficacy trial. If we better understand the successes and challenges of implementation, we may improve the overall effectiveness of combination, structural economic empowerment interventions for low-income communities. As such, Figure 1 depicts the conceptual model for the combined intervention.",
        'Acceptability Of A Microfinance-Based Empowerment Intervention For Transgender And Cisgender Women Sex Workers In Greater Kuala Lumpur, Malaysia': "Sample characteristics, stratified by gender identity, are reported in Table 2. Participants were on average 41 years old (SD = 12); and were mostly of Malay (42%), Indian (42%), or Chinese (9%) ethnicity. The majority of participants identified as Muslim (55%), followed by Hindu (27%), Buddhist (9%), Christian (6%), or Sikh (3%). Many participants reported not completing secondary school (45%) and less than half (46%) had children. Participants' demographic traits differed according to their gender identity, with a substantially higher proportion of TWSWs reporting being single (87%), having no children (87%), and holding some form of secondary schooling qualification (60%) when compared to CWSW. Moreover, a larger percentage of CWSWs reported being widowed/divorced (72%) in comparison to their TWSW counterparts.  Table 3 reports participants' patterns of income generation, stratified by gender identity. Participants reported having engaged in sex work for an average of 20 years (SD = 12); although, both CWSWs (61%) and TWSWs (72%) reported receiving income from non-sex work forms of work, including cleaning services, entertainment, and night club promotion. Participants' total mean monthly income was MYR 2235 (USD 521). Table 4 shows rates of lifetime drug use, stratified by gender identity. The most commonly used drugs were amphetaminetype substances (ATS) (50%), primarily crystal methamphetamine, followed by cannabis (36%) and heroin (27%). Engagement in drug use differed according to respondents' gender identity; with a higher percentage of CWSWs using ATS (61%) and/or heroine (44%) in comparison to TWSWs.\n\nThe in-depth interviews yielded three main themes regarding acceptability of a microfinance intervention: (a) participants were eager to engage in additional forms of income generation due to familial concerns and career aspirations; (b) interest in all proposed components of the intervention were driven by a desire to build their own business; and (c) potential challenges to developing businesses included a lack of financial resources, competition from other businesses, and fear of stigma."
            }
    analyzer.run(all_external_content, content_by_title)
