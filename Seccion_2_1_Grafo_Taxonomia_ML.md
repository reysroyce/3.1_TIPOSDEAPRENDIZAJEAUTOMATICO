# Actividad Integradora 3.1

## Tipos de aprendizaje automático

### Sección 2. Modelados --- 2.1 Grafo de taxonomía del Machine Learning (ML)

> **Tema:** Del dato al bien social: modelos de IA para transformar
> realidades\
> **Apartado:** 2.1 --- Clasificación jerárquica de los tipos y subtipos
> de aprendizaje automático

------------------------------------------------------------------------

## 2.1 Grafo de taxonomía del ML

### Introducción

La taxonomía del aprendizaje automático (*Machine Learning* o ML) es una
forma de organizar jerárquicamente sus principales enfoques, tareas y
algoritmos. En el nivel superior se encuentra el aprendizaje automático;
después se distinguen los tipos de aprendizaje; posteriormente se
identifican las tareas que resuelve cada enfoque y, finalmente, algunos
algoritmos que pueden utilizarse para realizarlas.

Esta organización permite comprender que un tipo de aprendizaje no es lo
mismo que una tarea ni que un algoritmo. Por ejemplo, el aprendizaje
supervisado es un enfoque, la clasificación es una tarea y K vecinos más
cercanos (KNN) es un algoritmo que puede utilizarse para resolverla
\[1\]\[2\].

### 2.1.1 Grafo jerárquico

El siguiente grafo presenta la estructura solicitada. Las relaciones
entre los nodos se indican mediante las expresiones **«es un tipo de»**,
**«resuelve»**, **«incluye»** y **«usa»**.

``` mermaid
flowchart TD
    ML["APRENDIZAJE AUTOMÁTICO (ML)"]

    ML -->|"incluye"| SUP["Aprendizaje supervisado"]
    ML -->|"incluye"| UNS["Aprendizaje no supervisado"]
    ML -->|"incluye"| REF["Aprendizaje por refuerzo"]
    ML -->|"incluye"| SEMI["Aprendizaje semisupervisado"]
    ML -->|"incluye"| TRANS["Aprendizaje por transferencia"]

    SUP -->|"resuelve"| CLAS["Clasificación"]
    SUP -->|"resuelve"| REG["Regresión"]
    CLAS -->|"usa"| KNN["K vecinos más cercanos (KNN)"]
    CLAS -->|"usa"| SVM["Máquinas de soporte vectorial (SVM)"]
    REG -->|"usa"| LIN["Regresión lineal"]
    REG -->|"usa"| ARB["Árboles de decisión"]

    UNS -->|"resuelve"| CLUST["Clustering o agrupamiento"]
    UNS -->|"resuelve"| RED["Reducción de dimensionalidad"]
    UNS -->|"resuelve"| ASOC["Asociación"]
    CLUST -->|"usa"| KMEANS["K-Means"]
    RED -->|"usa"| PCA["Análisis de componentes principales (PCA)"]
    ASOC -->|"usa"| APR["Apriori"]

    REF -->|"resuelve"| SEC["Toma secuencial de decisiones"]
    SEC -->|"usa"| QL["Q-Learning"]
    SEC -->|"usa"| SARSA["SARSA"]

    SEMI -->|"resuelve"| PSEUDO["Clasificación con pocas etiquetas"]
    PSEUDO -->|"usa"| PSEUDOALG["Pseudoetiquetado (self-training)"]

    TRANS -->|"incluye"| ADAPT["Adaptación de un modelo preentrenado"]
    ADAPT -->|"usa"| FT["Ajuste fino (fine-tuning)"]
```

**Cómo interpretar el grafo:**

-   El nodo raíz representa el concepto general: aprendizaje automático.
-   El primer nivel contiene los cinco enfoques principales incluidos en
    esta actividad.
-   El segundo nivel muestra tareas o estrategias que se realizan con
    cada enfoque.
-   El tercer nivel presenta algoritmos o técnicas representativas que
    pueden utilizarse para resolver esas tareas.
-   Las flechas indican la relación entre los elementos. Por ejemplo, el
    aprendizaje supervisado resuelve tareas de clasificación y
    regresión; la clasificación puede utilizar KNN o SVM.

> **Aclaración:** el aprendizaje por transferencia es una estrategia
> para reutilizar conocimiento de un modelo previamente entrenado. No es
> necesariamente una categoría excluyente o del mismo nivel conceptual
> que los enfoques supervisado, no supervisado y por refuerzo. Se
> incluye en el primer nivel porque forma parte de la clasificación
> solicitada en la actividad \[2\]\[3\].

------------------------------------------------------------------------

## 2.1.2 Descripción de los nodos de la taxonomía

### Nodo raíz: Aprendizaje automático

Es el área de la inteligencia artificial que desarrolla modelos capaces
de aprender patrones a partir de datos y utilizar lo aprendido para
hacer predicciones, identificar estructuras o tomar decisiones. Su
finalidad es resolver tareas sin tener que programar manualmente cada
regla para todos los casos \[1\]\[2\].

### Nodos de primer nivel: tipos de aprendizaje

  -----------------------------------------------------------------------
  Tipo                    Descripción             Característica
                                                  principal
  ----------------------- ----------------------- -----------------------
  Supervisado             Aprende con ejemplos    Compara sus
                          que incluyen entradas y predicciones con las
                          respuestas correctas    respuestas conocidas.
                          (etiquetas).            

  No supervisado          Analiza datos sin       Descubre grupos,
                          etiquetas para          relaciones o
                          encontrar estructuras o características
                          patrones.               relevantes.

  Por refuerzo            Aprende al interactuar  Mejora sus acciones
                          con un entorno y        mediante la
                          recibir recompensas o   experiencia.
                          penalizaciones.         

  Semisupervisado         Combina una cantidad    Aprovecha ambos
                          limitada de datos       conjuntos para entrenar
                          etiquetados con una     el modelo.
                          cantidad mayor de datos 
                          sin etiquetar.          

  Por transferencia       Reutiliza conocimiento  Adapta un modelo
                          de un modelo            existente a un nuevo
                          previamente entrenado   problema.
                          para una tarea          
                          relacionada.            
  -----------------------------------------------------------------------

### Nodos de segundo nivel: tareas y estrategias

  -----------------------------------------------------------------------
  Tipo de aprendizaje     Tarea o estrategia      ¿Qué realiza?
  ----------------------- ----------------------- -----------------------
  Supervisado             Clasificación           Asigna una categoría a
                                                  cada ejemplo, como
                                                  «spam» o «no spam».

  Supervisado             Regresión               Estima un valor
                                                  numérico, como el
                                                  precio de una vivienda.

  No supervisado          Clustering              Agrupa elementos con
                                                  características
                                                  similares.

  No supervisado          Reducción de            Reduce el número de
                          dimensionalidad         variables procurando
                                                  conservar información
                                                  importante.

  No supervisado          Asociación              Encuentra elementos o
                                                  eventos que aparecen
                                                  relacionados con
                                                  frecuencia.

  Por refuerzo            Toma secuencial de      Selecciona acciones
                          decisiones              considerando sus
                                                  consecuencias y
                                                  recompensas.

  Semisupervisado         Clasificación con pocas Aprende con pocos
                          etiquetas               ejemplos etiquetados y
                                                  más ejemplos sin
                                                  etiquetar.

  Por transferencia       Adaptación de un modelo Ajusta un modelo
                          preentrenado            existente para realizar
                                                  una tarea nueva
                                                  relacionada.
  -----------------------------------------------------------------------

### Nodos de tercer nivel: algoritmos y técnicas

  -----------------------------------------------------------------------
  Algoritmo o técnica     Tipo de aprendizaje     Función principal
                          relacionado             
  ----------------------- ----------------------- -----------------------
  KNN (K vecinos más      Supervisado             Predice una categoría o
  cercanos)                                       valor considerando los
                                                  ejemplos cercanos.

  SVM (máquinas de        Supervisado             Busca una separación
  soporte vectorial)                              adecuada entre clases;
                                                  también puede
                                                  utilizarse en
                                                  regresión.

  Regresión lineal        Supervisado             Modela la relación
                                                  entre variables para
                                                  estimar valores
                                                  numéricos.

  Árboles de decisión     Supervisado             Realizan predicciones
                                                  mediante reglas
                                                  organizadas en forma de
                                                  árbol.

  K-Means                 No supervisado          Agrupa datos en un
                                                  número definido de
                                                  grupos.

  PCA (análisis de        No supervisado          Reduce dimensionalidad
  componentes                                     mediante componentes
  principales)                                    que resumen la
                                                  variación de los datos.

  Apriori                 No supervisado          Encuentra reglas de
                                                  asociación entre
                                                  elementos frecuentes.

  Q-Learning              Por refuerzo            Aprende valores de
                                                  utilidad para elegir
                                                  acciones.

  SARSA                   Por refuerzo            Aprende una política a
                                                  partir de las acciones
                                                  realizadas y sus
                                                  recompensas.

  Pseudoetiquetado        Semisupervisado         Utiliza predicciones
  (*self-training*)                               del modelo para generar
                                                  etiquetas provisionales
                                                  de datos sin etiquetar.

  Ajuste fino             Transferencia           Continúa el
  (*fine-tuning*)                                 entrenamiento de un
                                                  modelo preentrenado con
                                                  datos de la nueva
                                                  tarea.
  -----------------------------------------------------------------------

**Importante:** los algoritmos no siempre pertenecen de manera exclusiva
a una sola tarea. Por ejemplo, SVM puede utilizarse tanto en
clasificación como en regresión. La tabla presenta sus usos
representativos para facilitar la comprensión de la taxonomía
\[2\]\[3\]\[4\].

------------------------------------------------------------------------

## 2.1.3 Ejemplo de lectura de la taxonomía

Un ejemplo sencillo es un sistema que identifica correos electrónicos no
deseados:

1.  **Nodo raíz:** pertenece al aprendizaje automático.
2.  **Primer nivel:** utiliza aprendizaje supervisado porque se entrena
    con correos previamente etiquetados.
3.  **Segundo nivel:** resuelve una tarea de clasificación, ya que
    asigna cada correo a una categoría.
4.  **Tercer nivel:** puede utilizar KNN o SVM como algoritmo de
    clasificación, dependiendo de los datos y del diseño del sistema.

Este recorrido permite distinguir los niveles de la taxonomía: el
enfoque indica cómo aprende el modelo, la tarea señala qué problema
resuelve y el algoritmo especifica un método para realizarlo.

------------------------------------------------------------------------

## Conclusión

El grafo de taxonomía permite organizar el aprendizaje automático desde
un concepto general hasta sus tipos, tareas y algoritmos. Esta
estructura ayuda a distinguir entre el enfoque de aprendizaje, el
problema que se desea resolver y el método que se utiliza para construir
el modelo.

También muestra que los diferentes enfoques tienen aplicaciones
distintas: el aprendizaje supervisado se utiliza en clasificación y
regresión; el no supervisado permite encontrar grupos y relaciones; el
aprendizaje por refuerzo se orienta a decisiones mediante recompensas;
el semisupervisado aprovecha datos etiquetados y no etiquetados; y el
aprendizaje por transferencia adapta conocimientos de modelos
existentes. Conocer estas diferencias facilita seleccionar una técnica
apropiada según el objetivo y los datos disponibles.

------------------------------------------------------------------------

## Referencias bibliográficas (formato APA 7)

\[1\] Google for Developers. (s. f.). *What is machine learning?*
https://developers.google.com/machine-learning/intro-to-ml/what-is-ml

\[2\] IBM. (s. f.). *What is machine learning?*
https://www.ibm.com/think/topics/machine-learning

\[3\] Mahesh, B. (2020). Machine learning algorithms---A review.
*International Journal of Science and Research (IJSR), 9*(1), 381--386.
https://pmc.ncbi.nlm.nih.gov/articles/PMC7983091/

\[4\] scikit-learn developers. (s. f.). *Supervised learning*.
https://scikit-learn.org/stable/supervised_learning.html

**Fecha de consulta:** 6 de noviembre de 2026.

------------------------------------------------------------------------

## Declaración de uso de inteligencia artificial

Para la elaboración de este apartado se utilizó ChatGPT (OpenAI) como
herramienta de apoyo para organizar la taxonomía, explicar las
relaciones entre tipos de aprendizaje, tareas y algoritmos, y mejorar la
redacción. Se incluyeron referencias bibliográficas para respaldar los
conceptos. El estudiante debe revisar y comprender el contenido antes de
presentarlo.

------------------------------------------------------------------------

**Fin del apartado 2.1 --- Grafo de taxonomía del Machine Learning.**
