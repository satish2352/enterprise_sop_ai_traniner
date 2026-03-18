import os
import sys

# Add the parent directory to sys.path so we can import multimodal_rag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from multimodal_rag import setup_query_engine, run_reflective_query

print("Starting test...")
query = "what is the process inside the screen?"
files = [f for f in os.listdir("./data") if f.endswith(".pdf")]
if not files:
    print("No pdfs to test with!")
else:
    test_file = files[0]
    print(f"Testing with file: {test_file}")
    
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="file_name", value=test_file)]
    )
    
    print("Running query...")
    try:
        response = run_reflective_query(query, filters=filters, top_k=5)
        print("--- RESPONSE ---")
        if isinstance(response, str):
            print(response)
        else:
            for t in response.response_gen:
                print(t, end="")
            print()
            print("\nSOURCES:")
            if hasattr(response, "source_nodes"):
                 for s in response.source_nodes:
                     print("-", s.node.metadata)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
