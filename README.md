# Classificação de Doenças em Folhas de Plantas (PlantVillage)

## Visão Geral do Projeto

Este projeto de Deep Learning tem como objetivo principal aplicar e comparar Redes Neurais Convolucionais (CNNs) para classificar imagens de folhas de plantas, identificando se estão saudáveis ou doentes. Utilizando o dataset **PlantVillage**, o projeto explora a criação de uma CNN do zero e a aplicação de técnicas de **Transfer Learning** (com o modelo MobileNetV2). O foco é comparar as duas abordagens para determinar qual oferece o melhor desempenho e generalização na identificação de doenças em plantas.

### Contexto e Objetivo

No agronegócio, a detecção precoce de doenças em culturas é crucial para garantir a segurança alimentar e reduzir perdas econômicas. Este projeto visa desenvolver um modelo de visão computacional preciso que possa auxiliar agricultores na identificação rápida de problemas em suas plantações. O objetivo é comparar uma arquitetura de CNN simples, treinada do zero, com um modelo de Transfer Learning (MobileNetV2) pré-treinado em um vasto conjunto de dados (ImageNet), avaliando qual abordagem é mais eficaz para este problema específico.

## Dataset

O dataset utilizado é o **PlantVillage** (disponível no Kaggle), que contém mais de 54.000 imagens de folhas de plantas. O conjunto de dados está organizado em **38 classes** (pastas), onde cada classe representa uma espécie de planta e sua condição (saudável ou um tipo específico de doença). As imagens são coloridas (RGB) e de alta qualidade.

## Análise Exploratória de Dados (EDA)

A EDA foi realizada para entender a estrutura do dataset. As principais etapas incluíram:

* Carregamento do dataset a partir dos diretórios.
* Contagem e listagem das classes, confirmando um total de **38 classes**.
* Visualização de imagens aleatórias de diferentes classes para inspecionar a variedade, iluminação e o desafio da classificação.

## Pré-processamento de Dados

Para preparar os dados para os modelos de Deep Learning, as seguintes etapas de pré-processamento foram executadas:

1.  **Divisão em Conjuntos de Treino e Validação:** O dataset foi dividido em **80% para treinamento** e **20% para validação** usando a função `image_dataset_from_directory`.
2.  **Redimensionamento de Imagens:** Todas as imagens foram padronizadas para o tamanho `(128, 128)` pixels.
3.  **Normalização:** Os valores dos pixels (originalmente de 0 a 255) foram normalizados para o intervalo `[0, 1]` usando uma camada `Rescaling(1./255)`.
4.  **Data Augmentation:** Para aumentar artificialmente o conjunto de dados de treino e prevenir overfitting, foram aplicadas transformações aleatórias em tempo real:
    * `RandomFlip("horizontal")`: Espelhamento horizontal.
    * `RandomRotation(0.2)`: Rotação aleatória de até 20%.
    * `RandomZoom(0.1)`: Zoom aleatório de até 10%.

## Modelagem e Avaliação

Dois modelos de CNN foram treinados e avaliados.

### 1. Modelo 1: CNN Simples (do Zero)

Uma Rede Neural Convolucional sequencial foi construída do zero com a seguinte arquitetura:

* `DataAugmentation` (camada de entrada)
* `Conv2D(32, (3,3), activation='relu')`
* `MaxPooling2D(2,2)`
* `Conv2D(64, (3,3), activation='relu')`
* `MaxPooling2D(2,2)`
* `Conv2D(128, (3,3), activation='relu')`
* `MaxPooling2D(2,2)`
* `Flatten()`
* `Dropout(0.5)`
* `Dense(128, activation='relu')`
* `Dense(38, activation='softmax')` (Camada de saída para 38 classes)

O modelo foi compilado com o otimizador `adam` e a função de perda `sparse_categorical_crossentropy`.

### 2. Modelo 2: Transfer Learning (MobileNetV2)

Utilizamos o modelo **MobileNetV2** (pré-treinado no ImageNet) como base, seguindo uma abordagem de duas fases:

* **Fase 1 (Feature Extraction):**
    1.  O `base_model` (MobileNetV2) foi carregado sem sua camada de classificação (`include_top=False`).
    2.  O `base_model` foi "congelado" (`trainable = False`).
    3.  Novas camadas de classificação foram adicionadas ao topo (`GlobalAveragePooling2D`, `Dropout(0.5)`, `Dense(128)`, e a `Dense(38, 'softmax')` final).
    4.  O modelo foi treinado por 10 épocas com um `Adam(learning_rate=0.001)`.

* **Fase 2 (Fine-Tuning):**
    1.  O `base_model` foi "descongelado" (`trainable = True`), exceto pelas primeiras 100 camadas.
    2.  O modelo foi re-compilado com uma taxa de aprendizado muito baixa (`Adam(learning_rate=1e-5)`) para ajustar sutilmente os pesos pré-treinados.
    3.  O treinamento continuou por mais 10 épocas.

## Resultados

Os modelos foram avaliados pela sua **Acurácia de Validação**. Os gráficos de Acurácia e Perda (Loss) ao longo das épocas foram plotados para analisar o aprendizado e o overfitting.

## Resultados
...
| Métrica | CNN Simples (do Zero) | MobileNetV2 (Fine-Tuning) |
| :--- | :---: | :---: |
| Acurácia (Validação) | **93.29%** | **95.34%** |
...

Os gráficos de treinamento do MobileNetV2 (Acurácia Total e Perda Total) mostram um aprendizado estável. A linha pontilhada indica o início do Fine-Tuning, onde a acurácia de validação geralmente recebe um impulso final e a perda de validação se estabiliza em um nível muito baixo.

## Conclusão

Ao comparar os resultados, o modelo de **Transfer Learning (MobileNetV2)** demonstrou um desempenho significativamente superior em termos de acurácia de validação e velocidade de convergência em comparação com a CNN simples construída do zero.

Isso era esperado, pois o MobileNetV2 já possui um conhecimento prévio robusto sobre como extrair features de imagens (bordas, texturas, formas), aprendido no dataset ImageNet. A CNN simples, por outro lado, precisa aprender essas features básicas do zero, o que exige muito mais dados e tempo de treino.

O Fine-Tuning se mostrou eficaz, ajustando os pesos do modelo pré-treinado para se especializarem na tarefa específica de classificação de folhas de plantas, resultando em um modelo final com alta precisão e boa generalização.

## Como Executar o Projeto

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/FelipeYacobian/Panda.git](https://github.com/FelipeYacobian/Panda.git)
    cd Panda
    ```

2.  **Ambiente de Desenvolvimento:**
    Este projeto foi desenvolvido utilizando **Google Colab**. Recomenda-se abrir o arquivo `.ipynb` no Google Colab para uma execução mais fácil.

3.  **Configurar o Kaggle:**
    * Para baixar o dataset, é necessário um arquivo `kaggle.json` (API token).
    * Faça o upload do seu `kaggle.json` quando o notebook solicitar (`files.upload()`).

4.  **Instale as Dependências (se estiver executando localmente):**
    ```bash
    pip install tensorflow numpy matplotlib kaggle
    ```

5.  **Execute o Notebook:**
    Abra o notebook e execute todas as células sequencialmente. O download do dataset (cerca de 800MB) e o treinamento dos modelos (especialmente o MobileNetV2) podem levar vários minutos.

## Tecnologias Utilizadas

* **Python**
* **TensorFlow** e **Keras**: Para construção, treinamento e avaliação das Redes Neurais.
* **MobileNetV2**: Arquitetura pré-treinada para Transfer Learning.
* **NumPy**: Para operações numéricas.
* **Matplotlib**: Para visualização de dados e gráficos de treinamento.
* **Kaggle API**: Para download do dataset.
* **Google Colab**: Ambiente de desenvolvimento (com GPU gratuita).

## Autores

* ViniMBlanco
* FelipeYacobian
