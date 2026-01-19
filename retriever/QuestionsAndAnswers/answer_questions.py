import os
import json
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv
import ast
import glob
from pathlib import Path  # Make sure Path is imported
from retriever.QuestionsAndAnswers.naiveQuestions import NaiveQuestions
from retriever.QuestionsAndAnswers.nuancedQuestions import PaperEmbeddingAnalyzer, NuancedQuestions
import re

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

class QuestionAnswerer:
    def __init__(self,  message_output=None):
        self.questions_list = []   
        self.relevant_papers_ids = [] 
        self.message_output = message_output or print

    def message(self, text):
        """
        Utility method to output messages
        """
        if self.message_output:
            self.message_output(text)

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
        
    def clean_json_string(self, json_string):
        retString = json_string.rstrip()
        if retString.endswith(")}"):
            retString = retString[:-2] + retString[-1]
        return retString
        
    def answer_question_gemini(self, questions, paper_text):
        
        genai.configure(api_key=gemini_api_key)
        # GEMINI SET UP
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 3,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # Generating the formatted list of questions
        formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

        # Creating the JSON schema
        schema = "paper_json = {\n"
        for question in questions:
            question_key = question.lower().replace(" ", "_").replace("?", "")
            schema += f'    "{question_key}": {{"type": "string"}},\n'
        schema = schema.rstrip(",\n") + "\n}"

        # Creating the prompt
        prompt = f"""STEP 1 - Answer the following questions based on the provided paper below. Respond using only the provided text, but adapt your answers to make them accessible to policymakers who might not be familiar with all technicalities. Use concise and detailed explanations, define technical terms and jargon in layman's terms and express dates in the format "Month, Year."


        {formatted_questions}

        STEP 2 - Using this JSON schema, return a JSON with the answers to the questions previously retrieved:
        {schema}

        If the provided paper contains no data to respond to a question, leave the field as an empty string and don't make up any data.
        """

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=prompt,
        )
        response = model.generate_content(paper_text)

        response_cleaned = self.clean_json_string(response.text)

        # Use json.loads to convert the cleaned string into a dictionary
        try:
            response_json = json.loads(response_cleaned)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {}

        return response_json
    
    def retrieve_naive(self, user_query): 
        print("Calling retrieve naive function")
        self.message("❓ Generating questions based on multiple perspectives on the topic provided to compare the papers retrieved ... ")
        # Run Naive Question class which identifies relevant papers and creates naive questions
        naive_questions = NaiveQuestions()
        self.relevant_papers_ids = naive_questions.run(user_query=user_query)

        # Find latest comparison question file and import it
        try:
            files = list(Path(".").glob("comparison_questions_*.txt"))
            
            if not files:
                print("No comparison questions files found.")
                return None
            
            # Sort files by modification time in descending order
            latest_questions_file = max(files, key=lambda f: f.stat().st_mtime)
            print(f"Found latest file: {latest_questions_file}")
        
        except Exception as e:
            print(f"Error accessing directory: {e}")
            return None
        
        # Transform txt into python list of questions
        try:
            with open(latest_questions_file, 'r', encoding='utf-8') as f:
                return  ast.literal_eval(f.read())
        except Exception as e:
            print(f"\nError reading questions file: {e}")
            return
        
    
    def generate_nuanced(self, external_contents, external_content_by_title):
        # This creates a file with nuanced for all relevant pappers 
        print("Generating nuanced questions.... ")
        self.message("🧐 Generating questions to capture the nuances of the retrieved papers ... ")
        embedding_analyzer = PaperEmbeddingAnalyzer()
        analyzer = NuancedQuestions(embedding_analyzer)
        analyzer.run(external_contents, external_content_by_title)
        return



    def retrieve_nuanced(self, paper_id):
        """
        Retrieves and parses questions for a given paper_id from the most recent question results file.
        """
        pattern = "question_results_*.jsonl"
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError("No question results files found.")
        
        latest_file = max(files, key=lambda x: x)
        
        try:
            with open(latest_file, 'r') as file:
                for line in file:
                    entry = json.loads(line.strip())
                    if entry.get('paper_id') == paper_id:
                        return entry['questions']
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON in file '{latest_file}': {str(e)}")
        
        # If no matching paper_id is found
        raise ValueError(f"Paper ID '{paper_id}' not found in {latest_file}")

    def run(self, user_query, all_external_content, external_content_by_title):
        print(f"\nStarting question answering script with naive and nuanced questions...")
        final_json = {}

        # Update questions list
        self.questions_list = self.retrieve_naive(user_query=user_query)

        # Safety check: if no papers were found, skip processing
        if not self.relevant_papers_ids:
            self.message("⚠️ No relevant papers found to analyze. Try a different query or adjust search parameters.")
            # Still save empty output
            output_path = os.path.join(os.getcwd(), "paper_answers.json")
            with open(output_path, "w") as json_file:
                json.dump({}, json_file, indent=4)
            return

        # Modify hereee
        self.generate_nuanced(all_external_content, external_content_by_title)

        self.message("📝 Starting to answer the questions generated for each relevant paper ...")
        # Answer questions for each paper
        for paper_id in self.relevant_papers_ids: 
            # If local paper
            if paper_id not in external_content_by_title.keys():
                paper_text = self.extract_text_from_pdf(paper_id)
                self.message("... ⏩️ Answering question for {} ...".format(paper_id))  

                # Retrieve nuanced questions
                nuanced = self.retrieve_nuanced(paper_id)

                all_questions = self.questions_list + nuanced

                # Call the function to get the answers for the questions
                answers = self.answer_question_gemini(all_questions, paper_text)
                
                # Add the answers to the final JSON dictionary under the paper_id
                final_json[paper_id] = answers

            # Else, paper from semantic scholar
            else: 
                print("\nReading extract from paper {} ...".format(paper_id))
                paper_text = external_content_by_title[paper_id]
                self.message("... ⏩️ Answering question for {} ...".format(paper_id))  
                print("Answering questions for {}".format(paper_id))
            
                # Retrieve nuanced questions
                nuanced = self.retrieve_nuanced(paper_id)

                all_questions = self.questions_list + nuanced

                # Call the function to get the answers for the questions
                answers = self.answer_question_gemini(all_questions, paper_text)
                
                # Add the answers to the final JSON dictionary under the paper_id
                final_json[paper_id] = answers

                if 'what_is_the_title_of_the_paper' not in final_json[paper_id] or not final_json[paper_id]['what_is_the_title_of_the_paper']:
                    final_json[paper_id]['what_is_the_title_of_the_paper'] = paper_id

        # Specify the filename and path to save the JSON
        output_path = os.path.join(os.getcwd(), "paper_answers.json")

        filtered_json = {}
        filtered_out = 0
        for paper_id, questions in final_json.items():
            # Be defensive: `questions` may be a dict (expected) or a list returned
            # by some upstream component. Handle both cases gracefully.
            if isinstance(questions, dict):
                answers_iter = questions.values()
            elif isinstance(questions, list):
                answers_list = []
                for item in questions:
                    if isinstance(item, dict):
                        answers_list.extend(item.values())
                    elif isinstance(item, str):
                        answers_list.append(item)
                answers_iter = answers_list
            else:
                answers_iter = []

            empty_count = sum(1 for answer in answers_iter if answer in (None, ""))
            if empty_count < 3:
                filtered_json[paper_id] = questions
            else:
                filtered_out += 1

        self.message(f"🔻 Filtered out {filtered_out} papers because they could not answer the generated comparison questions. ")
        print("Filtered out ", filtered_out, "papers because of empty repsonses")

        # Save the filtered JSON
        output_path = os.path.join(os.getcwd(), "paper_answers.json")

        with open(output_path, "w") as json_file:
            json.dump(filtered_json, json_file, indent=4)

        print(f"Output JSON saved at {output_path}")


if __name__ == "__main__":

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
    user_query = "I am the mayor of SF and I want to create a policy that fosters financial inclusion on the mission district. I want to implement this from a gender perspective focused on women that are substance users"
    
    answer = QuestionAnswerer()
    answer.run(user_query=user_query, all_external_content= all_external_content, external_content_by_title =  content_by_title)
