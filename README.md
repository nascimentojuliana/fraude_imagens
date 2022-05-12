odonto-fraud-classifier
==============================

Consta de uma biblioteca que utiliza dois modelos integrados em um pipeline para identificar se uma imagem está fraudada. 

* modelos devem ser treinados com imagens fraudadas e não fraudadas

O primeiro modelo separa as imagens de acordo com seu tipo (se é imagem de documento ou outras)

O segundo modelo:

1.uma rede xception pré-treinada, realizando fine-tunning para aprendizado da rede. 

Batch: utilizado para predições em batch com entrada em formato csv. O resultado final é retornado na forma de um score com a probabilidade da imagem ter sido fraudada antes de ser enviado pelo prestador.

Online: utilizado para processo de predição online, ou seja, basta o path da imagem que se quer analisar. O resultado final é a imagem com as áreas suspeitas de adulteração marcadas.

Para utilizar a biblioteca:

O pipeline online é carregado da seguinte forma:

1. Batch

1.1. RNA

pipeline = Pipeline(method='RGB', mode='xception',  dimension=299)
predictions = pipeline.predict(df, limiar1=0.5, limiar2=0.9)

Sendo method o tipo de método que será aplicado a imagem, utilizando a imagem RGB, ou com
técnicas de pre-processamento como ELA, ou PCA.
Mode é o modelo utilizado para fine-tunning, como inception ou xception.
Dimension é a dimesão da imagem, que para inception é 299.

O repositório utiliza códigos de outro repositório: https://github.com/GuidoBartoli/sherloq
Esse repositório contém a implementação de várias técnicas que são utilizadas por
peritos forenses em análise de imagens e foi adaptado para poder processar as imagens
antes de entrarem na rede neural.

Vocẽ pode usar esses códigos para treinar suas redes neurais de análise de fraude em imagens.