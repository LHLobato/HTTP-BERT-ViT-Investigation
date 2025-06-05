# Examinação de Vision Transformers aplicados em HTTP requests
Esse repositório é referente à aplicação de aprendizado de máquina voltado para visão computacional, com ênfase em Vision Transformers. 

#Métodologias de Aplicação

##Dastaset
O dataset utilizado nesse projeto foi o CSIC 2010, uma base de dados composta por HTTP requests. As amostras foram pré-processadas, de modo que componentes como métodos 'GET' 'POST', payload e URL foram mantidos, e convertidos em amostras textuais unificadas, como pode ser visto nos arquivos '.txt'. A partir desse procedimento, foi possível a utilização de alguns tokenizadores (modelos responsáveis por extrair representações numéricas a partir de textos), de modo a extrair características pertinentes das amostras analisadas em questão. 

##Reshaping
A partir do dataset pré-processado, tokenizadores foram aplicados, de modo a extraírem 4096 features. Neste projeto foram utilizados o TfidfVectorizer, do scikit-learn e o modelo Large BERT, da google. As saídas do procedimento de extração de features foram então redimensionadas, a partir de um PCA (scikit-learn), para 64 dimensões. A partir disso, métodos de reshaping comumente utilizados em séries temporais, como Gramian Angular Field (GASF e GADF) e Plot de Recorrência, foram investigados, formalizando imagens de 64 x 64 pixeis. 

##Treinamento
Foram examinados modelos Vision Transformers pré-treinados do ImageNet, de modo que foram examinadas diferentes situações de Transfer Learning e Fine-Tuning, variando o número de camadas descongeladas para atualização de pesos durante o treinamento.

#Conclusão
O desempenho apresentado mostrou-se viável e considerável na tarefa examinada, porém com possíveis melhorias. Os resultados podem ser encontrados na pasta dataframes do repositório, no arquivo test_https0.csv.

