# NLP Folder

[influence_arc_*.csv] :
| Field         | Type         | Null | Key | Default | Extra |
|---------------|--------------|------|-----|---------|-------|
| citingpaperID | varchar(100) | NO   | MUL | NULL    |       |
| citedpaperID  | varchar(100) | NO   | MUL | NULL    |       |
| sharedAuthor  | int          | NO   |     | 0       |       |
| oldProb       | float        | YES  |     | NULL    |       |

#oldProb(this attribute is deprecated, should use edge_extend_proba.csv)

[papers_arc_*.csv] :
| Field           | Type          | Null | Key | Default | Extra |
|-----------------|---------------|------|-----|---------|-------|
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


[top_field_authors.csv] : Top authors (ranked by H-index) information for each field.

[all_features.csv] : All features of the edges (citation links), used to calculate the extend probability of the edge in our experiment.



# OthersField

[links_*.csv] :
| Field         | Type         | Null | Key | Default | Extra |
|---------------|--------------|------|-----|---------|-------|
| citingpaperID | varchar(100) | NO   | MUL | NULL    |       |
| citedpaperID  | varchar(100) | NO   | MUL | NULL    |       |
| sharedAuthor  | int          | NO   |     | 0       |       |
| oldProb       | float        | YES  |     | NULL    |       |

#oldProb(this attribute is deprecated, should use edge_extend_proba.csv)

[papers_*.csv] :
| Field           | Type          | Null | Key | Default | Extra |
|-----------------|---------------|------|-----|---------|-------|
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


[top_field_authors.csv] : Top authors (ranked by H-index) information for each field.

[all_features.csv] : All features of the edges (citation links), used to calculate the extend probability of the edge in our experiment.