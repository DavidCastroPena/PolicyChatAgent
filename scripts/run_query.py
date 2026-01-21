import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retriever.multi_source_retriever import build_query_expansions, smart_retrieve

queries = [
    'I want to reduce poverty in Colombia',
    'poverty reduction Colombia'
]

for q in queries:
    print('\n' + '='*80)
    print('User query:', q)
    be = build_query_expansions(q)
    print('Expanded queries:')
    for i, x in enumerate(be['queries'],1):
        print(f'  {i}. {x}')
    # Run retrieval (limited to 30 results)
    try:
        papers = smart_retrieve(be['queries'], max_results=30)
    except Exception as e:
        print('Retrieval error:', e)
        papers = []
    print(f'Retrieved papers: {len(papers)}')
    titles = [p.get('title','') for p in papers[:10]]
    print('Top titles:')
    for t in titles:
        print('  -', t)
