from keybert import KeyBERT

doc="""O aprendizado automático (português brasileiro) ou a aprendizagem automática (português europeu) ou também aprendizado de máquina (português brasileiro) ou aprendizagem de máquina (português europeu) (em inglês: machine learning) é um subcampo da Engenharia e da ciência da computação que evoluiu do estudo de reconhecimento de padrões e da teoria do aprendizado computacional em inteligência artificial[1]. Em 1959, Arthur Samuel definiu aprendizado de máquina como o "campo de estudo que dá aos computadores a habilidade de aprender sem serem explicitamente programados"[2](livre tradução). O aprendizado automático explora o estudo e construção de algoritmos que podem aprender de seus erros e fazer previsões sobre dados[3]. Tais algoritmos operam construindo um modelo a partir de inputs amostrais a fim de fazer previsões ou decisões guiadas pelos dados ao invés de simplesmente seguindo inflexíveis e estáticas instruções programadas. Enquanto que na inteligência artificial existem dois tipos de raciocínio (o indutivo, que extrai regras e padrões de grandes conjuntos de dados, e o dedutivo), o aprendizado de máquina só se preocupa com o indutivo."""

#paraphrase-xlm-r-multilingual-v1
#bert-base-multilingual-cased

model = KeyBERT('bert-base-multilingual-cased')
keywords = model.extract_keywords(doc)

model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)

model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)

