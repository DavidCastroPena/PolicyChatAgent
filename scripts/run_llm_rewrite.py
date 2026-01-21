import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.analyze_query import generate_query_variants

cases = [
    'I want to reduce poverty in Colombia',
    'poverty reduction Colombia'
]

for c in cases:
    print('\n' + '='*60)
    print('Question:', c)
    variants = generate_query_variants(c, n=8, use_llm=True)
    print('Variants (rule-based):')
    for v in variants:
        print(' -', v)
    print('\n(If you want LLM rewrites, set use_llm=True and ensure API key is configured)')
