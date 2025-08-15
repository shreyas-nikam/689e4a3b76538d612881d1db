"""import streamlit as st

def calculate_context_relevancy(query, context):
    """Calculates the Context Relevancy metric."""
    if not query or not context:
        return 0.0

    query_words = query.lower().split()
    context_words = context.lower().split()

    if not query_words or not context_words:
        return 0.0

    common_words = set(query_words) & set(context_words)

    relevancy = len(common_words) / len(query_words)

    return relevancy

def calculate_groundedness(answer, context):
    """Calculates the Groundedness metric."""

    if not answer or not context:
        return 0.0

    answer_words = answer.lower().split()
    context_words = context.lower().split()

    if not answer_words or not context_words:
        return 0.0

    common_words = set(answer_words) & set(context_words)

    if not common_words:
        return 0.0

    return len(common_words) / len(answer_words)


def calculate_completeness(context, answer):
    """Calculates the Completeness metric."""
    if not context and not answer:
        return 1.0
    if not context:
        return 0.0
    if not answer:
        return 0.0

    context_words = context.lower().split()
    answer_words = answer.lower().split()

    if not context_words and not answer_words:
        return 1.0

    if not answer_words:
        return 0.0

    common_words = set(context_words) & set(answer_words)
    completeness = len(common_words) / len(answer_words) if answer_words else 0.0
    return completeness

def calculate_answer_relevancy(answer, query):
    """Calculates the Answer Relevancy metric."""

    if not answer or not query:
        return 0.0

    answer_words = answer.lower().split()
    query_words = query.lower().split()

    common_words = set(answer_words) & set(query_words)

    relevancy = len(common_words) / len(query_words) if len(query_words) > 0 else 0.0

    return min(1.0, relevancy)


def run_evaluation_metrics():
    st.header(\"Evaluation Metrics\")
    st.markdown(\"\"\"In this section, evaluate the quality of search results using embedding-based metrics.\"\"\")

    query = st.text_area(\"Query:\", value=\"\")
    context = st.text_area(\"Context:\", value=\"\")
    answer = st.text_area(\"Answer:\", value=\"\")

    if query and context and answer:
        context_relevancy = calculate_context_relevancy(query, context)
        groundedness = calculate_groundedness(answer, context)
        completeness = calculate_completeness(context, answer)
        answer_relevancy = calculate_answer_relevancy(answer, query)

        st.subheader(\"Metrics\")
        st.write(f\"Context Relevancy: {context_relevancy:.2f}\")
        st.write(f\"Groundedness: {groundedness:.2f}\")
        st.write(f\"Completeness: {completeness:.2f}\")
        st.write(f\"Answer Relevancy: {answer_relevancy:.2f}\")


"""