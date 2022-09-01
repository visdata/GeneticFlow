==============NLP==============
[influence_arc_*.csv] :
+---------------+--------------+------+-----+---------+-------+
| citingpaperID | varchar(100) | NO   | MUL | NULL    |       |
| citedpaperID  | varchar(100) | NO   | MUL | NULL    |       |
| sharedAuthor  | int          | NO   |     | 0       |       |
| oldProb       | float        | YES  |     | NULL    |       |
+---------------+--------------+------+-----+---------+-------+
#oldProb(this attribute is deprecated, should use edge_extend_proba.csv)

[papers_arc_*.csv] :
+-----------------+---------------+------+-----+---------+-------+
| paperID         | varchar(100)  | NO   | MUL | NULL    |       |
| title           | varchar(2000) | YES  |     | NULL    |       |
| year            | int           | YES  |     | NULL    |       |
| referenceCount  | int           | NO   |     | 0       |       |
| citationCount   | int           | NO   |     | 0       |       |
| authorOrder     | int           | YES  |     | NULL    |       |
| isCorePaper     | float         | YES  |     | NULL    |       |
| firstAuthorID   | varchar(15)   | YES  |     | NULL    |       |
| firstAuthorName | varchar(999)  | YES  |     | NULL    |       |
| venue           | varchar(999)  | YES  |     | NULL    |       |
| isACLPaper      | int           | YES  |     | NULL    |       |
| TopicLocation-X1| float         | YES  |     | NULL    |       |
| TopicLocation-X2| float         | YES  |     | NULL    |       |
+-----------------+---------------+------+-----+---------+-------+

[edge_extend_proba.csv] : The probability of the extend-type edge predicted by our model.

[top_field_authors.csv] : Top authors (ranked by citaions) information for each field.

[all_features.csv] : All features of the edges (citation links).



==============OthersField==============
[links_*.csv] :
+---------------+--------------+------+-----+---------+-------+
| citingpaperID | varchar(100) | NO   | MUL | NULL    |       |
| citedpaperID  | varchar(100) | NO   | MUL | NULL    |       |
| sharedAuthor  | int          | NO   |     | 0       |       |
| oldProb       | float        | YES  |     | NULL    |       |
+---------------+--------------+------+-----+---------+-------+
#oldProb(this attribute is deprecated, should use edge_extend_proba.csv)

[papers_*.csv] :
+-----------------+---------------+------+-----+---------+-------+
| paperID         | varchar(100)  | NO   | MUL | NULL    |       |
| title           | varchar(2000) | YES  |     | NULL    |       |
| year            | int           | YES  |     | NULL    |       |
| referenceCount  | int           | YES  |     | NULL    |       |
| citationCount   | int           | YES  |     | NULL    |       |
| authorOrder     | int           | YES  |     | NULL    |       |
| isKeyPaper      | float         | YES  |     | NULL    |       |
| firstAuthorID   | varchar(15)   | YES  |     | NULL    |       |
| firstAuthorName | varchar(999)  | YES  |     | NULL    |       |
| TopicLocation-X1| float         | YES  |     | NULL    |       |
| TopicLocation-X2| float         | YES  |     | NULL    |       |
+-----------------+---------------+------+-----+---------+-------+


[edge_extend_proba.csv] : The probability of the extend-type edge predicted by our model.

[top_field_authors.csv] : Top authors (ranked by citaions) information for each field.

[all_features.csv] : All features of the edges (citation links).