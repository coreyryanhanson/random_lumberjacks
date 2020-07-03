# Inserts escape characters in SQL query
def escape(cell):
    if type(cell) == str:
        if '"' in cell:
            return cell.replace('"', '')
    return cell

def escape_list(cell):
    if type(cell) == list:
        return [escape(item) for item in cell]
    return escape(cell)

# Creates a custom SQL insert statement based on a list of arguments.
def sql_query_generator(df, table_name, args):
    query = f"""INSERT INTO {table_name}
        """
    columns = """(
        """
    for j, arg in enumerate(args):
        if j < len(args) - 1:
            columns += arg + """,
        """
        else:
            columns += arg + """
        )
            """
    selection = ""
    for l in range(df.shape[0]):
        line = """(
                """
        for i, arg in enumerate(args):
            if i < len(args) - 1:
                line += '"' + str(escape_list(df[arg][l])) + """",
                """
            else:
                line += '"' + str(escape_list(df[arg][l])) + """"
            )"""
        if l < df.shape[0] - 1:
            selection += line + """,
            """
        else:
            selection += line
    return query + columns + "VALUES" + selection + """
         ;"""


# Inserts values from a dataframe using specific arguments to the specified table
def sql_insertion(df, table_name, db, args):
    cnx = mysql.connector.connect(
        host=config.aws_login.host,
        user=config.aws_login.user,
        passwd=config.aws_login.passwd,
        port=config.aws_login.port,
        database=db
    )
    print(cnx)
    cursor = cnx.cursor()
    query = sql_query_generator(df, table_name, args)
    cursor.execute(query)
    cursor.close()
    cnx.commit()
    cnx.close()