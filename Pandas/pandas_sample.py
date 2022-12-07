import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
Extract rows by gender.
'''
def get_rows_by_gender(df, gender):
    if (gender.lower() == 'male'):
        return df[df['Gender'] == 'M']
    elif (gender.lower() == 'female'):
        return df[df['Gender'] == 'F']
    return None


'''
Extract top salary for male and female.
'''
def get_top_salaries_for_both_genders(df):
    # get top salary for male
    males = df[df['Gender'] == 'M']
    idx = males['Salary'].argmax()  # index of max salary
    top_male = males.iloc[[idx]]    # get row by index
    # only want these 3 columns
    top_male = top_male.loc[:,['Name', 'Salary', 'Gender']]

    # get top salary for female
    females = df[df['Gender'] == 'F']
    idx = females['Salary'].argmax()
    top_female = females.iloc[[idx]]
    top_female = top_female.loc[:,['Name', 'Salary', 'Gender']]

    # combine into a single dataframe. 'ignore_index=True' re-orders (starts 
    # from 0) the indexes for the new dataframe
    return pd.concat([top_male, top_female], ignore_index=True)
    

'''
Get the average age and salary of all staff.
'''
def get_avg_age_n_salary(df):
    avg_age = int(df['Age'].mean())
    avg_salary = int(df['Salary'].mean())
    return (avg_age, avg_salary)


'''
Get staff having salaries above the company's average salary.
'''
def get_staff_with_above_avg_salary(df):
    avg_salary = int(df['Salary'].mean())
    # extract relevant rows using boolean selection
    above_avg_salary = df[df['Salary'] > avg_salary]
    # only want 'Name' and 'Salary' columns
    above_avg_salary = above_avg_salary.loc[:, ['Name', 'Salary']]
    return (avg_salary, above_avg_salary)


'''
Plot the no. of staff in each department.
'''
def plot_staff_by_dept(df):
    df2 = df.groupby('Department')['Name'].count()
    sns.barplot(x=df2.index, y=df2.values)
    plt.title('Staff Strength By Department')
    plt.show()


'''
Plot the no. of staff by gender.
'''
def plot_staff_by_gender(df):
    sns.countplot(data=df, x='Gender')
    plt.title('Staff Strength By Gender')
    plt.show()


'''
Plot the gender ratio of each department.
'''
def plot_depts_by_gender(df):    
    sns.displot(data=df, x='Department', hue='Gender', multiple='stack')
    plt.title('Departments by Gender', y=0.97) # 'y' to adjust title-position
    plt.show()

    
    

'''
Manipulating Pandas DataFrame in Python
'''

# display all staff
df = pd.read_csv('staff.csv')
print(df, '\n')

# display only male staff
males = get_rows_by_gender(df, 'male')
print(males, '\n')

# display only female staff
females = get_rows_by_gender(df, 'female')
print(females, '\n')

# display top salaries 
top_salaries = get_top_salaries_for_both_genders(df)
print(top_salaries, '\n')

# display staff's average age and salary
age, salary = get_avg_age_n_salary(df)
print('Average: Age={0}, Salary=${1}'.format(age, salary), '\n')

# display staff with above-avage salary
avg_salary, above_avg_salary = get_staff_with_above_avg_salary(df)
print('Average Salary: ${0}'.format(avg_salary))
print('Staff with above-avg salary:')
print(above_avg_salary)

plot_staff_by_dept(df)
plot_staff_by_gender(df)
plot_depts_by_gender(df)

