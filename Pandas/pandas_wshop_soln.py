import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
Find all staff with salary that is less than a certain amount.
'''
def staff_with_salary_less_than(df, salary):
    return df[df['Salary'] < salary]


'''
List all the staff for a given department.
'''
def list_staff(df, department):
    return df[df['Department'] == department]


'''
List the department with the highest number of staff.
'''
def department_with_most_staff(df):
    by_department = df.groupby(['Department'])['Department'].count()
    return (by_department.max(), by_department.idxmax())


'''
Plot staffs' salary in each department
'''
def plot_salary_for_each_department(df):
    # returns a Pandas Series
    by_dept_salary = df.groupby(['Department'])['Salary'].sum()

    # the 'index' gives us the Department names
    # the 'values' gives us the total sum of salary in each department
    sns.barplot(x=by_dept_salary.index, y=by_dept_salary.values)
    plt.show()


 
'''
Main program
'''

# read in data file 
df = pd.read_csv('staff.csv')
print(df, '\n')

# display staff with have salaries less a certain amount
print(staff_with_salary_less_than(df, 6000), '\n')

# display all the staff in a certain department
print(list_staff(df, 'Sales'), '\n')

# display the department with the most staff
staff_ct, department = department_with_most_staff(df)
print("'{0}' is the largest department with {1} staff".format(
    department, staff_ct), '\n')

# plot the total staffs' salary of each department
plot_salary_for_each_department(df)

