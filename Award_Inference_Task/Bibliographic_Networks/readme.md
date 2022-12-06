# Folder

[BiblioCouplingNetwork]:
[links_*.csv] :
| Field                  | Type         | Null | Key | Default | Extra |
| ---------------------- | ------------ | ---- | --- | ------- | ----- |
| paperID                | varchar(100) | YES  | MUL | NULL    |       |
| couplingpaperID        | varchar(100) | YES  | MUL | NULL    |       |
| fractional_couplingcnt | float        | YES  |     | NULL    |       |

[papers_*.csv] :
| Field         | Type         | Null | Key | Default | Extra |
|---------------|--------------|------|-----|---------|-------
| paperID       | varchar(100) | NO   | MUL | NULL    |       |
| year          | int          | YES  |     | NULL    |       |
| citationCount | int          | YES  |     | NULL    |       |
| authorOrder   | int          | YES  |     | NULL    |       |


[coauthorNetwork]:
[coauthors_*.csv] :
| Field               | Type         | Null | Key | Default | Extra |
|---------------------|--------------|------|-----|---------|-------|
| authorID            | varchar(15)  | YES  | MUL | NULL    |       |
| name                | varchar(999) | YES  |     | NULL    |       |
| authorRank          | int          | YES  |     | NULL    |       |
| PaperCount_field    | int          | YES  |     | NULL    |       |
| CitationCount_field | int          | YES  |     | NULL    |       |
| hIndex_field        | int          | YES  |     | NULL    |       |

[links_*.csv] :
| Field             | Type        | Null | Key | Default | Extra |
|-------------------|-------------|------|-----|---------|-------|
| authorID          | varchar(15) | YES  | MUL | NULL    |       |
| coauthorID        | varchar(15) | YES  | MUL | NULL    |       |
| coauthorpaperscnt | int         | YES  |     | NULL    |       |


[cocitationNetwork]:
[links_*.csv] :
| Field               | Type         | Null | Key | Default | Extra |
|---------------------|--------------|------|-----|---------|-------|
| paperID             | varchar(100) | YES  | MUL | NULL    |       |
| cocitpaperID        | varchar(100) | YES  | MUL | NULL    |       |
| fractional_cocitcnt | float        | YES  |     | NULL    |       |


[papers_*.csv] :
| Field         | Type         | Null | Key | Default | Extra |
|---------------|--------------|------|-----|---------|-------|
| paperID       | varchar(100) | NO   | MUL | NULL    |       |
| year          | int          | YES  |     | NULL    |       |
| citationCount | int          | YES  |     | NULL    |       |
| authorOrder   | int          | YES  |     | NULL    |       |

