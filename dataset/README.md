# Dataset de Hospitales

Tiene 5 columnas delimitadas por `|` que corresponden a `topic_id|pid|title|abstract|rel|sr_title`

* `topic_id`: id de la pregunta médica, es mas bien un tema médico representado como el **titulo de una revision sistematica**

* `pid`: Cantidad de articulos candidatos relacionados a la pregunta

* `title`: titulo del articulo

* `abstract`: abstract del articulo 

* `rel`: dice si el articulo es relevante (1) o no relevante (0) para la pregunta (topic_id) 

* `sr_title`: es el titulo del tema medico