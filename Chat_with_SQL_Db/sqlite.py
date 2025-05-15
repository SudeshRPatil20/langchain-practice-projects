import sqlite3

#connect to sqlite
connection=sqlite3.connect("student.db")

#create table by sqlite3
cursor=connection.cursor()

table_info="""
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

cursor.execute('''Insert into STUDENT values('Sudesh', 'Data Science', 'A', 86)''')
cursor.execute('''Insert into STUDENT values('Suket', 'Machine Learning', 'A', 89)''')
cursor.execute('''Insert into STUDENT values('Sam', 'Data Science', 'B', 98)''')
cursor.execute('''Insert into STUDENT values('Rahul', 'Artificial Intillegence', 'B', 98)''')
cursor.execute('''Insert into STUDENT values('Sumit', 'Data Science', 'A', 86)''')
cursor.execute('''Insert into STUDENT values('KAKA', 'MACHINE LEARNING', 'B', 99)''')

#DISPLAY THE TABLE
print("the given data is :-")
data=cursor.execute('''Select * from STUDENT''')
for row in data :
    print(row)
    
connection.commit()
connection.commit()