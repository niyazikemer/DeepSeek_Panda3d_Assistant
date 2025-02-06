from langchain_text_splitters import RecursiveCharacterTextSplitter

test_text = """
But if you want to do anything else --- for instance, if you wanted to do
painterly rendering, or water reflections, or lens flare, or ... well, your
imagination's the limit --- in that case, you need to write your own shaders.

.. toctree::
   :maxdepth: 2

   shader-basics
   list-of-glsl-inputs
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language='rst',
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(test_text)
for chunk in chunks:
    print("---CHUNK---")
    print(chunk)