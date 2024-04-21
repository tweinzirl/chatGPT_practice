### Overview
 - RAG: A technique for grounding LLM responses on *validated* external information.
 - Contextual integration: RAG apps load context into a database and then retrieve content for insertion into a prompt.
 - Documents contain elements (title, narrative text, lists, tables, images); useful document metadata includes filename, filetype, page number, section.
 - Extracting document data can be hard because of different formats and cues for element types (e.g., markdown vs html); different formats may require different extraction approaches (forms vs journal articles). Extracting metadata may depend on understanding document structure.


### Normalizing Content
 - First step is to convert all documents to a common format.
 - Second step is to serialize as JSON: common and well understood, is a standard HTTP response - Examples [Lesson_2_student.ipynb](Lesson_2_student.ipynb) use the unstructured API, which requires its own API key.

### Metadata and chunking
 - Metadata can include information about the source (filename, url, filetype, page number) and about the structure (section, element type and hierarchy).
 - Metadata can enhance search during RAG.
 - Semantic search steps:
  - input text
  - convert to embeddings
  - search vector database for relevant matches
  - insert relevant matches into prompt template
  - call LLM with prompt
 - [Notebook](Lesson_3_Student.ipynb) includes examples of using ChromaDB directly instead of through langchain
 - When creating VectorDB, chunking helps with the retrieval process. Chunking can be done logically based on document elements; for example, create a new chunk everying a section title is found. Smaller chunks can be combined so that all chunks are roughly the same size.

### PDFs and Images
 - Wheras HTML, Word, and Markdown include formatting information compatible with rule-based parsing, PDFs and Images provide only visual information.
 - Document Image Analysis (DIA) methods:
  - Document Layout Detection (DLD): uses object detection model to draw and label bounding boxes around layout elements on a document image.
  - Vision Transformers: Take a document image and produce structured text (JSON) as output. See [DONUT](https://arxiv.org/pdf/2111.15664.pdf) model. Pros: More flexible for non-standard document types like forms; more adaptible to new ontologies. Cons: Models are generative and prone to hallucination and repitition; computationally expensive.
  - DLD:
   - Use a computer vision model to identify and make a computer vision model ([YOLOX](https://arxiv.org/pdf/2107.08430.pdf) or Dectectron2)
   - Extract text from the box using **object character recognition**, when necessary. For some document types like PDF, using object character recognition is unnecessary.
   - Pros: Has a fixed set of element types and provides bounding box information.
   - Cons: Requires two model calls (for bounding box and object character recognition) and is less flexible.
 
### Tables
 - Tables are widely used in documents, especially in industries such as finance and insurance.
 - HTML and Word documents include table structure information. For other documents, it is necessary to infer the table information.
 - Techniques: [Table Transformers](https://arxiv.org/pdf/2203.01017.pdf), Vision Transformers, Optical Character Recognition postprocessing. Output of these techniques is html.
 - Table Transformer: Identify the table with a bounding box; then route the table to the Table Transformer. Pro: Trace cell back to the original bounding box. Con: Multiple expensive model calls
 - Vision Transformer: Outputs to html for table (JSON for text). Pros: allows for prompting, more flexible, one model call. Cons: Generative and prone to hallucination; no bounding box information.
 - OCR: OCR the table then infer the structure from patterns in the OCR output. Pros: fast and accurate for well-behaved tables. Cons: less flexible, no bounding box, requires rules-based parsing
 - Table extraction available as model-based API call from Unstructured

### Chatbot
 - Load data from multiple file types into a vectordb. Build a langchain chain (load_qa_with_sources_chain) to answer questions and return the source of the information.
