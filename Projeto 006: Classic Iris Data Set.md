# Projeto 6: Classic Iris Data Set.

Este é talvez o banco de dados mais conhecido encontrado na literatura sobre reconhecimento de padrões. O papel de Fisher é um clássico
no campo e é referenciado com frequência até hoje. (Veja Duda & Hart, por exemplo.) O conjunto de dados contém 3 classes de 50 instâncias
cada, onde cada classe se refere a um tipo de planta da íris. Uma classe é linearmente separável das outras duas; os últimos 
NÃO são linearmente separáveis um do outro.

Atributo previsto: classe da planta da íris.

Este é um domínio extremamente simples.
Estes dados diferem dos dados apresentados no artigo de Fishers (identificado por Steve Chadwick, spchadwick '@' espeedaz.net). 
A 35ª amostra deve ser: 4.9.3.1.1.5,0.2, "Iris-setosa", onde o erro está no quarto recurso. A 38ª amostra: 4.9.3.6.1.4.0.1, "Iris-setosa",
onde os erros estão na segunda e terceira características.

### Import packages and data
```{python, cache=FALSE, message=FALSE, warning=FALSE}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv("../input/Iris.csv")
print(iris.shape)
```
