# Predição de Depósito Bancário

## Visão Geral do Projeto
Este projeto de Deep Learning tem como objetivo principal aplicar e comparar Redes Neurais Convolucionais (CNNs) para classificar imagens de folhas de plantas, identificando se estão saudáveis ou doentes. Utilizando o dataset PlantVillage, o projeto explora a criação de uma CNN do zero e a aplicação de técnicas de Transfer Learning (com o modelo MobileNetV2). O foco é comparar as duas abordagens para determinar qual oferece o melhor desempenho e generalização na identificação de doenças em plantas.

## Contexto e Objetivo
No agronegócio, a detecção precoce de doenças em culturas é crucial para garantir a segurança alimentar e reduzir perdas econômicas. Este projeto visa desenvolver um modelo de visão computacional preciso que possa auxiliar agricultores na identificação rápida de problemas em suas plantações. O objetivo é comparar uma arquitetura de CNN simples, treinada do zero, com um modelo de Transfer Learning (MobileNetV2) pré-treinado em um vasto conjunto de dados (ImageNet), avaliando qual abordagem é mais eficaz para este problema específico.

## Dataset
O dataset utilizado é o PlantVillage (disponível no Kaggle), que contém mais de 54.000 imagens de folhas de plantas. O conjunto de dados está organizado em 38 classes (pastas), onde cada classe representa uma espécie de planta e sua condição (saudável ou um tipo específico de doença). As imagens são coloridas (RGB) e de alta qualidade.
