fraude_imagens
==============================

Consta de uma biblioteca que utiliza modelos integrados em um pipeline para identificar se uma imagem está fraudada. 

* modelos devem ser treinados com imagens fraudadas e não fraudadas

1.uma rede xception pré-treinada, realizando fine-tunning para aprendizado da rede. 
2.métodos de tratamento da imagem segundo técnicas forenses

neural_network tem a classe RNA, que deve ser chamada para fazer o treinamento, teste e predict dos modelos.

pre_processing tem a classe PreProcessing, que processa a imagem segundo o método forense escolhido.

O scrip train.py utiliza como entrada um csv com path das imagens, sua classificação e uso (train, test, validate).

O scrip train.py utiliza como entrada um csv com path das imagens, sua classificação e uso (test)

O scrip predict utiliza como entrada um csv com path das imagens e retorno um csv com o path das imagens e 
sua classificação em 0 ou 1, sendo 1 com limiar de 0.9.

O repositório utiliza códigos de outro repositório: https://github.com/GuidoBartoli/sherloq
Esse repositório contém a implementação de várias técnicas que são utilizadas por
peritos forenses em análise de imagens e foi adaptado para poder processar as imagens
antes de entrarem na rede neural.

Vocẽ pode usar esses códigos para treinar suas redes neurais de análise de fraude em imagens.
