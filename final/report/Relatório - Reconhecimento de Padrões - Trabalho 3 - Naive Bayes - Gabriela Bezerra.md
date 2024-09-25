![[header.png]]

# Relatório - Reconhecimento de Padrões 2024.1
## Trabalho 3 - Naive Bayes
*14/08/2024*
- Professor: Ajalmar Rêgo da Rocha Neto, Dr
- Aluna: Gabriela de Carvalho Barros Bezerra
- Código: [Repositório do Github](https://github.com/GabrielaBezerra/pattern-recognition)

</br>

# 1. Introdução

Este trabalho consistiu na implementação de um Classificador Naive Bayes, aplicação deste nos cojuntos Iris, Coluna 2D, Coluna 3D, Artificial I, Artificial II, Dermatology e Breast Cancer. Também foi realizado o cálculo das métricas de acurácia e desvio padrão das taxas de acerto para cada conjunto de dados. Além disso, as métricas encontradas foram comparadas às dos modelos anteriormente implementados (KNN, DMC, e Classificador Bayesiano Multivariado).

Foram feitos 28 experimentos: um para cada conjunto de dados (Iris, Coluna 2D, Coluna 3D, Artificial I, Artificial II, Dermatology, Breast Cancer) utilizando cada modelo (KNN, DMC, Classificador Bayesiano Multivariado e Naive Bayes). Cada experimento consistiu em 20 realizações de uma simulação computacional que se deu em 6 fases:

1. Leitura do *conjunto de dados*.
2. Preprocessamento do *conjunto de dados*.
3. Separação do *conjunto de dados* em *conjunto de treinamento* e *conjunto de teste*.
4. Treinamento do *modelo* com o *conjunto de treinamento*.
5. Predição do *conjunto de teste* utilizando o *modelo*.
6. Cáculo das *métricas*. 

Como resultado, foram obtidas visualizações dos conjuntos de treinamento e teste para cada *conjunto de dados*, visualizações das *superfícies de decisão* e a *densidade de probabilidade* para cada par de características de todos os conjuntos de dados (somente um par de cada exemplo mostrado no relatório), as métricas de *acurácia* e *desvio padrão* para cada modelo em cada *conjunto de dados*, e as *matrizes de confusão* da pior realização de cada modelo em cada *conjunto de dados*.

# 2. Metodologia

Durante a implementação do trabalho, foram utilizadas as bibliotecas:

- *pandas* para leitura e separação dos *conjuntos de dados* (fases 1 e 2) e para construção da matriz de confusão (fase 5).
- *numpy* para realização de cálculos e manipulações vetorais (fases 3, 4, e 5).
- *matplotlib* para visualização dos conjuntos de treinamento e teste (fase 2) e exibição das superfícies de decisão no experimentos adicionais.

## 2.1. Implementação dos Modelos

Na implementação do trabalho, cada modelo foi abstraído como uma classe com métodos *fit* para treinamento no *conjunto de treinamento* e *predict* para predição do *conjunto de teste*. O modelo KNN foi implementado com o hiperparâmetro K, estimado através do cálculo da raiz quadrada da quantidade de pontos no conjunto de dados utilizado no treinamento. O modelo DMC não foi implementado com hiperparâmetros. Para ambos modelos, foi utilizada a distância euclidiana. Em ambos classificadores bayesianos foram utilizadas a distribuição gaussiana, também conhecida como distribuição normal. O classifcador bayesiano multivariado utiliza a distribuição gaussiana multivariada, calculada a partir de matrizes de covariância. O classificador naive bayes utiliza a distribuição gaussiana univariada, calculada a partir da simples variância (desvio padrão ao quadrado), e partindo do pressuposto "inocente" (naive) de que os atributos da base são independentes entre si.

## 2.2. Experimentos principais

### 2.2.1. Leitura dos conjuntos de dados

A leitura do *conjunto de dados* foi feita utilizando a função *read_csv* da biblioteca *pandas*. Durante as visualizações, foi identificado e removido um *outlier* nos conjuntos de dados Coluna 2D e Coluna 3D.

### 2.2.2. Preprocessamento dos conjuntos de dados

Após a leitura do *conjunto de dados*, foi realizado um preprocessamento nos dados em três passos:

1. Foram removidas as linhas que não tinham todas as colunas preenchidas com dados
2. Dados textuais/categóricos foram transformados em dados numéricos
3. Foram removidas possíveis linhas duplicadas

Esse preprocessamento foi essencial para que os dados estivem em um formato computável pelos modelos.

### 2.2.3 Separação do conjunto de dados em treinamento e teste

Para separação do *conjunto de dados*, foi utilizado um método conhecido como *Holdout*, que embaralha os pontos de maneira aleatória e divide o conjunto de dados proporcionalmente em conjuntos de treinamento e teste. O embaralhamento foi implementado utilizando a função *sample* da biblioteca *pandas*. Foi parametrizado uma porcentagem para a divisão do conjunto de dados em os conjuntos de treinamento e teste. Por padrão, foi utilizado 80% para *conjunto de treinamento* e 20% para *conjunto de teste*.

Visualização gráfica após dos conjuntos de treinamento e teste após a separação do *conjunto de dados* Artificial I em uma realização.
![[Screenshot 2024-04-03 at 22.41.58.png|500]]

Visualização gráfica após dos conjuntos de treinamento e teste após a separação do *conjunto de dados* Artificial II em uma realização.
![[Screenshot 2024-04-16 at 21.52.45.png|500]]

Visualização gráfica após dos conjuntos de treinamento e teste após a separação do *conjunto de dados* Iris em uma realização.
![[Screenshot 2024-04-03 at 22.45.03.png|500]]

Visualização gráfica após dos conjuntos de treinamento e teste após a separação do *conjunto de dados* Coluna 2D em uma realização.
![[Screenshot 2024-04-16 at 22.27.48.png|500]]

Visualização gráfica após dos conjuntos de treinamento e teste após a separação do *conjunto de dados* Coluna 3D em uma realização.
![[Screenshot 2024-04-16 at 22.28.44.png|500]]

Visualização gráfica após dos conjuntos de treinamento e teste após a separação do *conjunto de dados* Dermatology em uma realização.
![[Screenshot 2024-04-16 at 22.31.07.png|500]]

Visualização gráfica após dos conjuntos de treinamento e teste após a separação do *conjunto de dados* Breast Cancer em uma realização.
![[Screenshot 2024-04-16 at 22.34.21.png|500]]


### 2.2.4. Treinamento do modelo

#### KNN

A implementação da função *fit* do modelo KNN consiste em simplesmente armazenar os pontos do *conjunto de treinamento* em memória.

#### DMC

Durante a implementação da função *fit*, o modelo DMC calcula os valores médios de cada característica para cada classe no conjunto de treinamento. Esses valores médios são denomidados *centróides*, e são armazenados para posterior uso na etapa de predição.

#### Bayes Multivariado

A implementação da função *fit* do modelo Bayesiano Multivariado consiste em calcular a probabilidade a priori, a média e a matriz de covariância associadas a cada classe. Esse valores serão utilizados durante a predição.

#### Naive Bayes

A implementação da função *fit* do modelo Naive Bayes consiste em calcular a probabilidade a priori, a média e a variância (desvio padrão ao quadrado) associadas a cada atributo, para cada classe. Esse valores serão utilizados durante a predição.

### 2.2.5. Predição do modelo

#### KNN
A implementação da função *predict* consiste em, para cada ponto do *conjunto de teste*:
1. Calcular a *distância euclidiana* entre o ponto de teste e todos os pontos de treinamento.
2. Ordenar as distâncias encontradas e identificar os k-vizinhos (pontos de treinamento) mais próximos.
3. Atribuir a classe mais frequente entre os k-vizinhos mais próximos ao ponto de teste.

Esse processo é repetido para todos os pontos do *conjunto de teste*, e as classes que resultam da predição são retornadas como saída.

#### DMC
A implementação da função *predict* consiste em, para cada ponto do conjunto de teste:

1. Calcular a *distância euclidiana* entre o ponto de teste e os *centróides* de cada classe previamente calculados durante o treinamento.
2. Identificar o *centróide* mais próximo.
3. Atribuir a classe do *centróide* mais próximo ao ponto de teste.

Esse processo é repetido para todos os pontos do *conjunto de teste*, e as classes que resultam da predição são retornadas como saída.

#### Bayes Multivariado

A função predict consiste em, para cada ponto do conjunto de teste:

1. Ler a probabilidade a priori, a média e a matriz de covariância associadas a cada classe calculadas na função *fit*.
2. Calcular a verossimilhança do ponto que está sendo testado sob a distribuição gaussiana da classe usando a função `_multivariate_gaussian` que utiliza uma equação exponencial não simplificada (como requisitado no trabalho). Nesse momento também foi aplicada a estratégia de regularização para permitir o cálculo da inversa da matriz de covariância, necessária à equação utilizada.
3. Multiplicar a priori da classe pela verossimilhança e armazenar o resultado (probabilidade posterior).
4. Escolher a classe com a maior probabilidade posterior como a previsão para o dado de teste atual.

Esse processo é repetido para todos os pontos do *conjunto de teste*, e as classes que resultam da predição são retornadas como saída.

#### Naive Bayes

A função predict consiste em, para cada ponto do conjunto de teste:

1. Ler a probabilidade a priori, a média e a matriz de covariância associadas a cada classe calculadas na função *fit*.
2. Calcular a verossimilhança do ponto que está sendo testado usando a função `_univariate_gaussian` que retorna o produtório das distribuições gaussianas univariadas de cada atributo. Nesse momento também foi aplicada a estratégia de regularização para evitar valores inválidos no denominador ou na raiz quadrada da equação.
3. Multiplicar a priori da classe pela verossimilhança e armazenar o resultado (probabilidade posterior).
4. Escolher a classe com a maior probabilidade posterior como a previsão para o dado de teste atual.

Esse processo é repetido para todos os pontos do *conjunto de teste*, e as classes que resultam da predição são retornadas como saída.

### 2.2.6. Cálculo das métricas

As métricas foram calculadas através da construção da *matriz de confusão* seguida do cômputo da *taxa de acerto* e *desvio padrão* a partir das predições realizadas. 

Ao final de cada experimento foi exibida a *matriz de confusão* para a realização que obteve as piores métricas (acurácia geral mais baixa), no intuíto de entender como o modelo está se comportando no pior caso.

## 2.3. Experimentos adicionais

Também foram realizados experimentos adicionais para cada par de características em cada conjunto de dados, e cada modelo. No melhor caso de cada experimento adicional, foi exibida a *superfície de decisão* 2D e 3D, e as gausianas das funções de densidade de probabilidade no intuído de entender qual par de características separa melhor cada conjunto de dados.

# 3. Resultados e Discussões

## 3.1 Conjunto de dados Artificial I

O conjunto de dados Artificial I é linearmente separável, como podemos ver pela superfície de decisão com as únicas duas características do *conjunto de dado* Artificial I para o KNN, DMC, BMC e NB. Podemos exibir as gaussianas em 3D do BMC e NB montadas a partir do calculo da densidade de probabilidade de cada classe versus um par de atributos.
![[Screenshot 2024-04-03 at 22.42.40.png|500]]
![[Screenshot 2024-04-03 at 22.44.35.png|500]]
==BMC superficie de decisao 2D
BMC gaussiana 3D
NB superficie de decisao 2D
NB gaussiana 2D==

**Métricas a partir das taxas de acerto das 20 realizações**

>**Acurácia**
>
>| Classe | KNN | DMC | BMC | NB | 
>|-|-|-|-|-|
>| All | 1.00 | 1.00 | 1.00 | |
>| 0.0 | 1.00 | 1.00 | 1.00 | |
> 1.0 | 1.00 | 1.00 | 1.00 | |

>**Desvio Padrão**
>
>| Classe | KNN | DMC | BMC | NB | 
>|-|-|-|-|-|
>| All | 0.00 | 0.00 | 0.00 | |
>| 0.0 | 0.00 | 0.00 | 0.00 | |
>| 1.0 | 0.00 | 0.00 | 0.00 | |

**Matriz de Confusão do KNN, DMC, BMC e NB**
```
True       0  1
Predicted      
0          4  0
1          0  8
```


## 3.2. Iris  

O conjunto de dados Iris parece ser linearmente separável, como podemos ver pela superfície de decisão com as duas características que melhor separam as classes do *conjuntos de dados* Iris para o KNN e DMC.
![[Screenshot 2024-04-03 at 22.45.21.png|500]]
![[Screenshot 2024-04-03 at 22.45.45.png|500]]

Como o BMC e o NB utilizam a distribuição normal, o conjunto de dados Iris foi separado com hiperboles aproximadamente centralizadas nos dados, como podemos ver pela superfície de decisão e gaussianas a partir da densidade de probabilidade construídas com as duas características que melhor separam as classes do *conjunto de dado* Iris para ambos os modelos.
==BMC superficie de decisao 2D
BMC gaussiana 3D
NB superficie de decisao 2D
NB gaussiana 2D==

**Métricas a partir das taxas de acerto das 20 realizações**

>**Acurácia**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.96 | 0.93 | 0.98 | |
>| Iris-setosa | 1.00 | 1.00 | 1.00 | |
>| Iris-versicolor | 0.94 | 0.89 | 0.99 | |
>| Iris-virginica | 0.93 | 0.92 | 0.95 | |

>**Desvio Padrão**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.03 | 0.03 | 0.02 | |
>| Iris-setosa | 0.00 | 0.00 | 0.00 | |
>| Iris-versicolor | 0.07 | 0.09 | 0.03 | |
>| Iris-virginica | 0.07 | 0.08 | 0.05 | |

**Matriz de Confusão KNN (realização com pior acurácia)**
```
Predicted        Iris-setosa  Iris-versicolor  Iris-virginica
True                                                         
Iris-setosa                6                0               0
Iris-versicolor            0                9               1
Iris-virginica             0                1              13
```

**Matriz de Confusão DMC (realização com pior acurácia)**
```
Predicted        Iris-setosa  Iris-versicolor  Iris-virginica
True                                                         
Iris-setosa               11                0               0
Iris-versicolor            0                8               1
Iris-virginica             0                2               8
```

**Matriz de Confusão BMC (realização com pior acurácia)**
```
True             Iris-setosa  Iris-versicolor  Iris-virginica
Predicted                                                    
Iris-setosa               21                0               0
Iris-versicolor            0               10               2
Iris-virginica             0                0              12
```

==**Matriz de Confusão NB (realização com pior acurácia)**==
```
True             Iris-setosa  Iris-versicolor  Iris-virginica
Predicted                                                    
Iris-setosa               x                0               0
Iris-versicolor            0               x               x
Iris-virginica             0                0              x
```

## 3.3. Coluna 2D

Superfície de decisão com as duas características que melhor separam as classes do *conjunto de dado* Coluna 2D para o KNN e DMC.
![[Screenshot 2024-04-03 at 22.46.58.png|500]]
![[Screenshot 2024-04-03 at 22.47.12.png|500]]

Superfície de decisão e gaussianas de densidade de probabilidade construídas com as duas características que melhor separam as classes do *conjunto de dado* Coluna 2D para o BMC e NB.
==BMC superficie de decisao 2D
BMC gaussiana 3D
NB superficie de decisao 2D
NB gaussiana 2D==

**Métricas a partir das taxas de acerto das 20 realizações**

>**Acurácia**
>
>| Classe | KNN | DMC | BMC | NB
>|-|-|-|-|-|
>| All | 0.85 | 0.77 | 0.83 | |
>| AB | 0.90 | 0.95 | 0.94 | |
>| NO | 0.77 | 0.60 | 0.69 ||

>**Desvio Padrão**
>
>| Classe | KNN | DMC | BMC | NB
>|-|-|-|-|-|
>| All | 0.03 | 0.05 | 0.03 | |
>| AB | 0.05 | 0.04 | 0.03 | |
>| NO | 0.11 | 0.08 | 0.06 | |

**Matriz de Confusão KNN (realização com pior acurácia)**
```
Predicted  AB  NO
True             
AB         39   4
NO          4  15
```

**Matriz de Confusão DMC (realização com pior acurácia)**
```
Predicted  AB  NO
True             
AB         29   2
NO         11  20
```

**Matriz de Confusão BMC (realização com pior acurácia)**
```
Worst Confusion Matrix:
True       AB  NO
Predicted        
AB         49  12
NO          1  31
```

==**Matriz de Confusão NB (realização com pior acurácia)**==
```
Worst Confusion Matrix:
True       AB  NO
Predicted        
AB          x  x
NO          x  x
```

## 3.4. Coluna 3D  

O conjunto de dados Coluna 3D não é um problema linearmente separável, portanto não foi possível encontrar uma superfície de decisão com as duas características que separam bem as classes do *conjunto de dado* Coluna 3D para o KNN e o DMC.
![[Screenshot 2024-04-03 at 22.49.50.png|500]]
![[Screenshot 2024-04-03 at 22.50.11.png|500]]

Superfície de decisão e gaussianas de densidade de probabilidade construídas com as duas características que melhor separam as classes do *conjuntos de dados* Coluna 3D para o BMC e NB.
![[Screenshot 2024-04-16 at 22.28.58.png|500]]
![[Screenshot 2024-04-16 at 22.26.36.png|500]]

**Métricas a partir das taxas de acerto das 20 realizações**

>**Acurácia**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.82 | 0.76 | 0.83 | |
>| DH | 0.76 | 0.55 | 0.68 | |
>| NO | 0.68 | 0.69 | 0.94 | |
>| SL | 0.97 | 0.97 | 0.77 | |

>**Desvio Padrão**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.04 | 0.05 | 0.03 | |
>| DH | 0.15 | 0.10 | 0.10 | |
>| NO | 0.09 | 0.06 | 0.03 | |
>| SL | 0.03 | 0.03 | 0.08 | |

**Matriz de Confusão KNN (realização com pior acurácia)**
```
Predicted  DH  NO  SL
True                 
DH         11   1   1
NO          6  18   2
SL          0   1  22
```

**Matriz de Confusão DMC (realização com pior acurácia)**
```
Predicted  DH  NO  SL
True                 
DH          9   7   1
NO          1  16   3
SL          0   2  23
```

**Matriz de Confusão BMC (realização com pior acurácia)**
```
True       DH  NO  SL
Predicted            
DH         12   0   5
NO          1  41   2
SL          4   2  26
```

==**Matriz de Confusão NB (realização com pior acurácia)**==
```
True       DH  NO  SL
Predicted            
DH          x   0   x
NO          x   x   x
SL          x   x   x
```

## 3.5. Artificial II

O conjunto de dados Artificial II é problema linearmente separável, abaixo está a superfície de decisão com as duas características que separam bem as classes do *conjunto de dado* Artificial II para o KNN, DMC, BMC e NB. Podemos exibir as gaussianas em 3D do BMC e NB montadas a partir do calculo da densidade de probabilidade de cada classe versus um par de atributos.
![[Screenshot 2024-04-17 at 00.33.10.png|500]]
![[Screenshot 2024-04-17 at 00.33.34.png|500]]
![[Screenshot 2024-04-16 at 21.53.06 1.png|500]]
![[Screenshot 2024-04-16 at 21.53.43 1.png|500]]

**Métricas a partir das taxas de acerto das 20 realizações**

>**Acurácia**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 1.00 | 1.00 | 0.99 | |
>| star | 1.00 | 1.00 | 1.00 | |
>| triangle | 1.00 | 1.00 | 0.97 | |
>| circle | 1.00 | 1.00 | 1.00 | |

>**Desvio Padrão**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.00 | 0.00 | 0.04 | |
>| star | 0.00 | 0.00 | 0.00 | |
>| triangle | 0.00 | 0.00 | 0.10 | |
>| circle | 0.00 | 0.00 | 0.00 | |

**Matriz de Confusão KNN (realização com pior acurácia)**
```
True        star  triangle  circle
Predicted               
star          4      0        0
triangle      0      2        0
circle        0      0        1
```

**Matriz de Confusão DMC (realização com pior acurácia)**
```
True        star  triangle  circle
Predicted               
star          3      0        0
triangle      0      2        0
circle        0      0        2
```

**Matriz de Confusão BMC (realização com pior acurácia)**
```
True        star  triangle  circle
Predicted               
star          4      0        0
triangle      1      1        0
circle        0      0        1
```

==**Matriz de Confusão NB (realização com pior acurácia)**==
```
True        star  triangle  circle
Predicted               
star          x      0        0
triangle      x      x        0
circle        0      0        x
```

## 3.6. Dermatology

**Métricas a partir das taxas de acerto das 20 realizações**

>**Acurácia**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.76 | 0.50 | 0.88 | |
>| 1 | 0.96 | 0.55 | 1.00 | |
>| 2 | 0.75 | 0.30 | 0.73 | |
>| 3 | 0.94 | 0.66 | 0.93 | |
>| 4 | 0.35 | 0.33 | 0.97 | |
>| 5 | 0.45 | 0.25 | 0.94 | |
>| 6 | 0.95 | 1.00 | 0.00 | |

>**Desvio Padrão**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.04 | 0.06 | 0.03 | |
>| 1 | 0.03 | 0.09 | 0.00 | |
>| 2 | 0.14 | 0.14 | 0.09 | |
>| 3 | 0.05 | 0.10 | 0.06 | |
>| 4 | 0.21 | 0.14 | 0.04 | |
>| 5 | 0.16 | 0.12 | 0.06 | |
>| 6 | 0.09 | 0.00 | 0.00 | |

**Matriz de Confusão KNN (realização com pior acurácia)**
```
True       1.0  2.0  3.0  4.0  5.0  6.0
Predicted                              
1.0         24    2    3    3    1    0
2.0          0   11    0    6    7    1
3.0          0    0   19    0    0    0
4.0          0    1    0    6    3    0
5.0          0    2    0    0    7    0
6.0          3    1    0    0    0    8
```

**Matriz de Confusão DMC (realização com pior acurácia)**
```
True       1.0  2.0  3.0  4.0  5.0  6.0
Predicted                              
1.0         12    0    0    0    0    0
2.0          5    4    9    3    4    0
3.0          7    9   13    5    5    0
4.0          1    2    0    3    0    0
5.0          0    1    0    2    1    0
6.0          4    1    0    2    4   11
```

**Matriz de Confusão BMC (realização com pior acurácia)**
```
True       1.0  2.0  3.0  4.0  5.0  6.0
Predicted                              
1.0         31    2    1    1    1    9
2.0          0   12    0    0    0    0
3.0          0    0   18    0    0    0
4.0          0    5    0   14    0    0
5.0          0    0    0    0   14    0
6.0          0    0    0    0    0    0
```

==**Matriz de Confusão NB (realização com pior acurácia)**==
```
True       1.0  2.0  3.0  4.0  5.0  6.0
Predicted                              
1.0          x    x    x    x    x    x
2.0          0    x    0    0    0    0
3.0          0    0    x    0    0    0
4.0          0    x    0    x    0    0
5.0          0    0    0    0    x    0
6.0          0    0    0    0    0    0
```

## 3.6. Breast Cancer

**Métricas a partir das taxas de acerto das 20 realizações**

>**Acurácia**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.75 | 0.72 | 0.75 | |
>| no | 0.92 | 0.76 | 0.86 | |
>| yes | 0.18 | 0.59 | 0.41 | |

>**Desvio Padrão**
>
>| Classe | KNN | DMC | BMC | NB |
>|-|-|-|-|-|
>| All | 0.04 | 0.04 | 0.03 | |
>| no | 0.05 | 0.08 | 0.04 | |
>| yes | 0.12 | 0.16 | 0.10 | |

**Matriz de Confusão KNN (realização com pior acurácia)**
```
True        no  yes
Predicted          
no          49   13
yes         12    5
```

**Matriz de Confusão DMC (realização com pior acurácia)**
```
True        no  yes
Predicted          
no          33    7
yes         24   15
```

**Matriz de Confusão BMC (realização com pior acurácia)**
```
True        no  yes
Predicted          
no          47   15
yes         10    7
```

**Matriz de Confusão NB (realização com pior acurácia)**
```
True        no  yes
Predicted          
no           x    x
yes          x    x
```

# 4. Conclusão

==| Base | Melhores Modelos pela acurácia média entre classes |
|-|-|
| Artificial I | KNN e DMC, Bayes |
| Artificial II | KNN e DMC, Bayes |
| Iris | Bayes, KNN, DMC |
| Coluna 2D | KNN, Bayes, DMC |
| Coluna 3D | Bayes, KNN, DMC|
| Dermatology | Bayes, KNN, DMC |
| Breast Cancer | Bayes e KNN*, DMC | 
*Para o *conjunto de dados* Breast Cancer, é importante notar que, apesar da acurácia média entre classes equiparável ao Bayes Multivariado, o KNN teve uma acurácia muito mais baixa para a classe "no", significando que quando há incerteza ele tende a classificar um novo dado como "yes". Com essas informações, a escolha do melhor classificador para resolver o problema depende das necessidades de quem utilizará.
O modelo Bayesiano Multivariado classificou bem todas as bases de dados, com destaque para problemas de classificação de mais de duas classes ou com muitas dimensões pois, com o uso função de densidade de probabilidade, foi possível separar melhor as classes existentes do que com o KNN e o DMC. As bases Dermatology e Breast Cancer precisaram de preprocessamento para serem utilizadas pois ambas contém dados categóricos e dados faltantes. Além disso, foi necessária aplicar a estratégia de regularização na implementação do modelo Bayesiano Multivariado e Naive Bayes para que fosse possível calcular a distribuição normal. Apesar de mudar infimamente os valores que vem dos dados, a utilização da regularização não afeta negativamente o classificador.==

##### Final Metrics for Artificial I

Accuracy

| |All|0|1|
|-|-|-|-|
|DMC|1.0|1.0|1.0|
|NaiveBayes|0.95|0.87|1.0|
|KNN|1.0|1.0|1.0|
|BayesianGaussianMultivariate|0.97|0.94|1.0|

Standard Deviation

| |All|0|1|
|-|-|-|-|
|DMC|0.0|0.0|0.0|
|NaiveBayes|0.08|0.22|0.0|
|KNN|0.0|0.0|0.0|
|BayesianGaussianMultivariate|0.07|0.16|0.0|

##### Final Metrics for Artificial II

Accuracy

| |All|star|triangle|circle|
|-|-|-|-|-|
|DMC|1.0|1.0|1.0|1.0|
|NaiveBayes|1.0|1.0|1.0|1.0|
|KNN|1.0|1.0|1.0|1.0|
|BayesianGaussianMultivariate|0.95|0.97|0.94|1.0|

Standard Deviation

| |All|star|triangle|circle|
|-|-|-|-|-|
|DMC|0.0|0.0|0.0|0.0|
|NaiveBayes|0.0|0.0|0.0|0.0|
|KNN|0.0|0.0|0.0|0.0|
|BayesianGaussianMultivariate|0.13|0.08|0.18|0.0|

##### Final Metrics for Iris

Accuracy

| |All|Iris-setosa|Iris-versicolor|Iris-virginica|
|-|-|-|-|-|
|DMC|0.91|1.0|0.90|0.84|
|NaiveBayes|0.94|1.0|0.92|0.90|
|KNN|0.94|1.0|0.92|0.93|
|BayesianGaussianMultivariate|0.96|1.0|0.93|0.97|

Standard Deviation

| |All|Iris-setosa|Iris-versicolor|Iris-virginica|
|-|-|-|-|-|
|DMC|0.04|0.0|0.08|0.11|
|NaiveBayes|0.02|0.0|0.07|0.06|
|KNN|0.03|0.0|0.06|0.08|
|BayesianGaussianMultivariate|0.03|0.0|0.07|0.03|

##### Final Metrics for Column 2D

Accuracy

| |All|AB|NO|
|-|-|-|-|
|DMC|0.75|0.67|0.93|
|NaiveBayes|0.77|0.74|0.84|
|KNN|0.84|0.85|0.81|
|BayesianGaussianMultivariate|0.82|0.79|0.89|

Standard Deviation

| |All|AB|NO|
|-|-|-|-|
|DMC|0.03|0.05|0.03|
|NaiveBayes|0.02|0.03|0.05|
|KNN|0.03|0.04|0.05|
|BayesianGaussianMultivariate|0.03|0.03|0.05|

##### Final Metrics for Column 3D

Accuracy

| |All|DH|SL|NO|
|-|-|-|-|-|
|DMC|0.77|0.78|0.65|0.85|
|NaiveBayes|0.81|0.67|0.66|0.98|
|KNN|0.82|0.49|0.83|0.95|
|BayesianGaussianMultivariate|0.83|0.60|0.77|0.97|

Standard Deviation

| |All|DH|SL|NO|
|-|-|-|-|-|
|DMC|0.04|0.10|0.08|0.04|
|NaiveBayes|0.03|0.06|0.09|0.01|
|KNN|0.03|0.17|0.05|0.02|
|BayesianGaussianMultivariate|0.02|0.07|0.06|0.01|

##### Final Metrics for Dermatology

Accuracy

| |All|1|2|3|4|5|6|
|-|-|-|-|-|-|-|-|
|DMC|0.51|0.57|0.32|0.67|0.32|0.30|0.99|
|NaiveBayes|0.87|1.0|0.40|0.95|0.98|0.94|0.83|
|KNN|0.76|0.94|0.74|0.90|0.38|0.46|0.92|
|BayesianGaussianMultivariate|0.87|1.0|0.74|0.94|0.97|0.94|0.0|

Standard Deviation

| |All|1|2|3|4|5|6|
|-|-|-|-|-|-|-|-|
|DMC|0.05|0.09|0.10|0.11|0.14|0.14|0.03|
|NaiveBayes|0.02|0.0|0.12|0.03|0.02|0.05|0.18|
|KNN|0.04|0.03|0.13|0.06|0.13|0.19|0.11|
|BayesianGaussianMultivariate|0.03|0.0|0.10|0.03|0.03|0.05|0.0|

##### Final Metrics for Breast Cancer

Accuracy

| |All|no|yes|
|-|-|-|-|
|DMC|0.73|0.76|0.62|
|NaiveBayes|0.77|0.85|0.53|
|KNN|0.76|0.93|0.14|
|BayesianGaussianMultivariate|0.75|0.85|0.42|

Standard Deviation

| |All|no|yes|
|-|-|-|-|
|DMC|0.05|0.06|0.10|
|NaiveBayes|0.05|0.04|0.11|
|KNN|0.04|0.04|0.09|
|BayesianGaussianMultivariate|0.06|0.04|0.12|