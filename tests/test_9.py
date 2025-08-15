import pytest
from definition_60b98260cd644b61b92d05eb23d8ba51 import calculate_answer_relevancy

@pytest.fixture
def mock_model():
    class MockModel:
        def encode(self, sentences):
            # Simple mock embedding for testing purposes
            if isinstance(sentences, str):
                if sentences == "What were the key takeaways from the earnings call regarding future investments?":
                    return [0.1, 0.2]
                elif sentences == "The earnings call emphasized strategic investments in AI research and development. The CEO also mentioned expanding into new geographical markets.":
                    return [0.3, 0.4]
                else:
                    return [0.0, 0.0]
            else:
                if sentences[0] == "What were the key takeaways from the earnings call regarding future investments?":
                    return [[0.1, 0.2]]
                elif sentences[0] == "The earnings call emphasized strategic investments in AI research and development.":
                    return [[0.3, 0.4]]
                elif sentences[1] == "The CEO also mentioned expanding into new geographical markets.":
                    return [[0.5, 0.6]]
                elif sentences[0] == "Some irrelevant sentence.":
                    return [[0.0, 0.0]]
                elif sentences[1] == "Another irrelevant sentence.":
                    return [[0.0, 0.0]]
                
                else:
                    return [[0.0, 0.0]]
    return MockModel()


def test_calculate_answer_relevancy_high(mock_model):
    query = "What were the key takeaways from the earnings call regarding future investments?"
    answer = "The earnings call emphasized strategic investments in AI research and development. The CEO also mentioned expanding into new geographical markets."
    
    # Mock implementation
    def mock_calculate_answer_relevancy(query, answer, model):
      query_embedding = model.encode(query)
      answer_embedding = model.encode(answer)

      
      
      query_sentences = [query]
      answer_sentences = [answer]

      query_embeddings = [model.encode(s) for s in query_sentences]
      answer_embeddings = [model.encode(s) for s in answer_sentences]

      similarities = []
      for a_emb in answer_embeddings:
          max_sim = max([cosine_similarity(a_emb, q_emb) for q_emb in query_embeddings])
          similarities.append(max_sim)

      return sum(similarities) / len(similarities)

    def cosine_similarity(v1, v2):
        dot_product = sum(x * y for x, y in zip(v1, v2))
        magnitude_v1 = sum(x ** 2 for x in v1) ** 0.5
        magnitude_v2 = sum(x ** 2 for x in v2) ** 0.5
        return dot_product / (magnitude_v1 * magnitude_v2)

    expected_relevancy = mock_calculate_answer_relevancy(query, answer, mock_model)

    # Replace the real function call with the mocked implementation in your test file
    from definition_60b98260cd644b61b92d05eb23d8ba51 import calculate_answer_relevancy 
    result = calculate_answer_relevancy(query, answer, mock_model)
    assert result == expected_relevancy

def test_calculate_answer_relevancy_low(mock_model):
    query = "What were the key takeaways from the earnings call regarding future investments?"
    answer = "Some irrelevant sentence. Another irrelevant sentence."
    
    # Mock implementation
    def mock_calculate_answer_relevancy(query, answer, model):
      query_embedding = model.encode(query)
      answer_embedding = model.encode(answer)

      
      
      query_sentences = [query]
      answer_sentences = [answer]

      query_embeddings = [model.encode(s) for s in query_sentences]
      answer_embeddings = [model.encode(s) for s in answer_sentences]

      similarities = []
      for a_emb in answer_embeddings:
          max_sim = max([cosine_similarity(a_emb, q_emb) for q_emb in query_embeddings])
          similarities.append(max_sim)

      return sum(similarities) / len(similarities)

    def cosine_similarity(v1, v2):
        dot_product = sum(x * y for x, y in zip(v1, v2))
        magnitude_v1 = sum(x ** 2 for x in v1) ** 0.5
        magnitude_v2 = sum(x ** 2 for x in v2) ** 0.5
        return dot_product / (magnitude_v1 * magnitude_v2)

    expected_relevancy = mock_calculate_answer_relevancy(query, answer, mock_model)

    # Replace the real function call with the mocked implementation in your test file
    from definition_60b98260cd644b61b92d05eb23d8ba51 import calculate_answer_relevancy 
    result = calculate_answer_relevancy(query, answer, mock_model)
    assert result == expected_relevancy

def test_calculate_answer_relevancy_empty_query(mock_model):
    query = ""
    answer = "The earnings call emphasized strategic investments in AI research and development. The CEO also mentioned expanding into new geographical markets."

    # Mock implementation
    def mock_calculate_answer_relevancy(query, answer, model):
      query_embedding = model.encode(query)
      answer_embedding = model.encode(answer)

      
      
      query_sentences = [query]
      answer_sentences = [answer]

      query_embeddings = [model.encode(s) for s in query_sentences]
      answer_embeddings = [model.encode(s) for s in answer_sentences]

      similarities = []
      for a_emb in answer_embeddings:
          max_sim = max([cosine_similarity(a_emb, q_emb) for q_emb in query_embeddings])
          similarities.append(max_sim)

      return sum(similarities) / len(similarities)

    def cosine_similarity(v1, v2):
        dot_product = sum(x * y for x, y in zip(v1, v2))
        magnitude_v1 = sum(x ** 2 for x in v1) ** 0.5
        magnitude_v2 = sum(x ** 2 for x in v2) ** 0.5
        return dot_product / (magnitude_v1 * magnitude_v2)

    expected_relevancy = mock_calculate_answer_relevancy(query, answer, mock_model)

    # Replace the real function call with the mocked implementation in your test file
    from definition_60b98260cd644b61b92d05eb23d8ba51 import calculate_answer_relevancy 
    result = calculate_answer_relevancy(query, answer, mock_model)
    assert result == expected_relevancy

def test_calculate_answer_relevancy_empty_answer(mock_model):
    query = "What were the key takeaways from the earnings call regarding future investments?"
    answer = ""

    # Mock implementation
    def mock_calculate_answer_relevancy(query, answer, model):
      query_embedding = model.encode(query)
      answer_embedding = model.encode(answer)

      
      
      query_sentences = [query]
      answer_sentences = [answer]

      query_embeddings = [model.encode(s) for s in query_sentences]
      answer_embeddings = [model.encode(s) for s in answer_sentences]

      similarities = []
      for a_emb in answer_embeddings:
          max_sim = max([cosine_similarity(a_emb, q_emb) for q_emb in query_embeddings])
          similarities.append(max_sim)

      return sum(similarities) / len(similarities)

    def cosine_similarity(v1, v2):
        dot_product = sum(x * y for x, y in zip(v1, v2))
        magnitude_v1 = sum(x ** 2 for x in v1) ** 0.5
        magnitude_v2 = sum(x ** 2 for x in v2) ** 0.5
        return dot_product / (magnitude_v1 * magnitude_v2)

    expected_relevancy = mock_calculate_answer_relevancy(query, answer, mock_model)

    # Replace the real function call with the mocked implementation in your test file
    from definition_60b98260cd644b61b92d05eb23d8ba51 import calculate_answer_relevancy 
    result = calculate_answer_relevancy(query, answer, mock_model)
    assert result == expected_relevancy

def test_calculate_answer_relevancy_same_query_answer(mock_model):
    query = "What were the key takeaways from the earnings call regarding future investments?"
    answer = "What were the key takeaways from the earnings call regarding future investments?"

    # Mock implementation
    def mock_calculate_answer_relevancy(query, answer, model):
      query_embedding = model.encode(query)
      answer_embedding = model.encode(answer)

      
      
      query_sentences = [query]
      answer_sentences = [answer]

      query_embeddings = [model.encode(s) for s in query_sentences]
      answer_embeddings = [model.encode(s) for s in answer_sentences]

      similarities = []
      for a_emb in answer_embeddings:
          max_sim = max([cosine_similarity(a_emb, q_emb) for q_emb in query_embeddings])
          similarities.append(max_sim)

      return sum(similarities) / len(similarities)

    def cosine_similarity(v1, v2):
        dot_product = sum(x * y for x, y in zip(v1, v2))
        magnitude_v1 = sum(x ** 2 for x in v1) ** 0.5
        magnitude_v2 = sum(x ** 2 for x in v2) ** 0.5
        return dot_product / (magnitude_v1 * magnitude_v2)

    expected_relevancy = mock_calculate_answer_relevancy(query, answer, mock_model)

    # Replace the real function call with the mocked implementation in your test file
    from definition_60b98260cd644b61b92d05eb23d8ba51 import calculate_answer_relevancy 
    result = calculate_answer_relevancy(query, answer, mock_model)
    assert result == expected_relevancy
