def GetTableNames(dbname):
    import win32com.client

    conn = win32com.client.Dispatch(r"ADODB.Connection")
    DSN = "PROVIDER=Microsoft.Jet.OLEDB.4.0;DATA SOURCE=%s;" % dbname
    conn.Open(DSN)
    oCat = win32com.client.Dispatch(r"ADOX.Catalog")
    oCat.ActiveConnection = conn
    oTab = oCat.Tables
    a = []
    for x in oTab:
        if x.Type == "TABLE":
            a.append(x.Name)
    conn.Close()
    return a


def GetFieldNames(dbname, tablename):
    import win32com.client

    conn = win32com.client.Dispatch(r"ADODB.Connection")
    DSN = "PROVIDER=Microsoft.Jet.OLEDB.4.0;DATA SOURCE=%s;" % dbname
    conn.Open(DSN)
    rs = win32com.client.Dispatch(r"ADODB.Recordset")
    rs.Open(tablename, conn, 1, 3)
    b = []
    for x in range(rs.Fields.Count):
        b.append(rs.Fields.Item(x).Name)
    conn.Close()
    return b


def ConnectDB(dbname):
    import pyodbc

    conn = pyodbc.connect(r"Driver={Microsoft Access Driver (*.mdb)};DBQ=" + dbname)
    return conn


def ConnectMySQLDB(host, port, db, usr, pwd):
    import pyodbc

    driver = (
        r"Driver={MySQL ODBC 8.0 Unicode Driver};Server="
        + host
        + ";Port="
        + port
        + ";Database="
        + db
        + ";User="
        + usr
        + "; Password="
        + pwd
        + ";Option=3;"
    )
    # print driver;
    conn = pyodbc.connect(driver)
    return conn
