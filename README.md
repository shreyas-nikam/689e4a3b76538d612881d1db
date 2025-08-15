# QuLab: Visual Embedding Lab for Financial Documents

## Project Title and Description

QuLab is a Streamlit application designed to explore and visualize semantic relationships within a corpus of financial documents using embedding techniques. This lab allows users to upload financial documents, visualize their embeddings in an interactive atlas, perform semantic searches, and evaluate retrieval quality using embedding-based metrics. This lab is designed for users who wish to understand how machine learning and natural language processing can be applied to analyze financial documents.

## Features

*   **Document Upload and Processing:**
    *   Upload multiple financial documents in PDF format.
    *   Extract and clean text from uploaded PDF files (lowercasing, removing extra spaces).
    *   Display processed documents in a DataFrame.

*   **Embedding Visualization:**
    *   Generate dense vector embeddings for each document using Sentence-BERT.
    *   Compute a 2D projection of text embeddings for interactive visualization.
    *   Display an interactive embedding atlas with document points.
    *   Select and view data points from the embedding atlas using DuckDB queries.

*   **Semantic Search:**
    *   Perform semantic similarity search using cosine similarity.
    *   Find the top-N most similar documents to a given query.
    *   Display search results with document IDs and similarity scores.

*   **Evaluation Metrics:**
    *   Calculate and display embedding-based evaluation metrics:
        *   Context Relevancy: Measures the relevance of the context to the query.
        *   Groundedness: Measures how well the answer is supported by the context.
        *   Completeness: Measures how completely the answer covers the context.
        *   Answer Relevancy: Measures the relevance of the answer to the query.

## Getting Started

### Prerequisites

*   Python 3.7+
*   Pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your_repository_url>
    cd QuLab
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt` file, you can create one or install the packages individually:

    ```bash
    pip install streamlit pandas pypdf sentence-transformers atlas-embed duckdb
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application in your browser:**
    Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Using the Application:**
    *   **Document Processing:**
        *   Upload financial documents in PDF format using the file uploader.
        *   Review the processed documents displayed in the DataFrame.
    *   **Embedding Visualization:**
        *   Explore the interactive embedding atlas to visualize document relationships.
        *   Use the predicate function to select a subset of the documents.
        *   Perform semantic searches by entering a query and specifying the number of results.
    *   **Evaluation Metrics:**
        *   Enter a query, context, and answer in the text areas.
        *   View the calculated evaluation metrics for the given inputs.

## Project Structure

```
QuLab/
├── app.py                         # Main Streamlit application file
├── application_pages/
│   ├── document_processing.py   # Document processing functionality
│   ├── embedding_visualization.py # Embedding visualization functionality
│   └── evaluation_metrics.py    # Evaluation metrics functionality
├── requirements.txt             # List of Python dependencies
├── README.md                      # Project documentation (this file)
└── LICENSE                       # Licensing information (if applicable)
```

## Technology Stack

*   **Streamlit:** Used for creating the interactive web application.
*   **Pandas:** Used for data manipulation and analysis (DataFrames).
*   **PyPDF:** Used for extracting text from PDF documents.
*   **Sentence Transformers:** Used for generating document embeddings.
*   **Atlas Embed:** Used for creating the 2D projection for Visualization.
*   **DuckDB:** Used for in-memory data analytics and SQL queries on DataFrames.
*   **Python:** The primary programming language.

## Contributing

Contributions are welcome!  Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE) (or specify the appropriate license).

## Contact

*   For questions or issues, please contact: [quantuniversity](https://www.quantuniversity.com/)
