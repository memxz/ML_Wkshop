import os
import pandas as pd


'''
Implement the Python functions below.
'''



'''
Find all staff with salary that is less than a certain amount.
'''
def staff_with_salary_less_than(df, salary):
    pass


'''
List all the staff for a given department.
'''
def list_staff(df, department):
    pass


'''
List the department with the highest number of staff.
'''
def department_with_most_staff(df):
    pass


'''
Plot staffs' salary in each department
'''
def plot_salary_for_each_department(df):
    pass



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
print("'{0}' is the biggest with {1} staff".format(
    department, staff_ct), '\n')

# plot the total staffs' salary of each department
plot_salary_for_each_department(df)
