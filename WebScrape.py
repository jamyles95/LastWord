
# coding: utf-8

# In[1]:


import urllib
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from bs4.element import Tag
from pandas import DataFrame
import time


# In[12]:


url = urllib.request.urlopen( 'http://www.tdcj.state.tx.us/death_row/dr_executed_offenders.html' )
doc = url.read()
tree = BeautifulSoup( doc,'html.parser' )
print(tree.prettify())


# In[14]:


body = tree.find('body')
#print(body)


# In[15]:


info = body.find_all('a',string=['Offender Information','Offender Information '])
statement = body.find_all('a',string=['Last Statement','Last Statement ','Last Statemen'])
info_links=[]
state_links=[]

#get links to offender information
for i in info:
    info_links.append(i.get('href'))
    
    
#get links to last statement    
for i in statement:
    state_links.append(i.get('href'))
#print(info)
#len(info_links)

print(len(state_links))
print(len(statement))
#info_links
#state_links


# In[14]:


# info[1].contents
# len(info)

# tr = body.find_all('tr')
# len(tr)
# a = body.find_all('a')
# a = a[1:len(a)]
# cont=[]
# for i in a:
#     cont.append(i.contents[0])
    
# cont = pd.Series(cont)

# cont.unique()

# stat=[]
# for i in state_links:
#     url_state = urllib.request.urlopen( 'https://www.tdcj.state.tx.us/death_row/' + i )
#     doc_state = url_state.read()
#     tree_state = BeautifulSoup( doc_state,'html.parser' )
#     body_state = tree_state.find('div',{'id':'body'})
#     p_laststate = body_state.find_all('p',{'class':"text_bold"})
#     for j in p_laststate:
#         stat.append(j.contents[0])
        
# stat = pd.Series(stat)
# stat.unique()


# In[16]:


#get first, last name, age, date, race, county
tr = body.find_all('tr')
tr = tr[1:len(tr)]
last_name = []
first_name = []
age = []
date=[]
race=[]
county=[]
for i in tr:
    last_name.append(i.find_all('td')[3].contents[0])
    first_name.append(i.find_all('td')[4].contents[0])
    age.append(i.find_all('td')[6].contents[0])
    date.append(i.find_all('td')[7].contents[0])
    race.append(i.find_all('td')[8].contents[0])
    county.append(i.find_all('td')[9].contents[0])


# In[18]:


#Testing
url_state = urllib.request.urlopen( 'https://www.tdcj.state.tx.us/death_row/'+state_links[542] )
doc_state = url_state.read()
tree_state = BeautifulSoup( doc_state,'html.parser' )
body_state = tree_state.find('body')
p = body_state.find_all('p')
s = ""
type(p[7].contents[0]) == Tag
type(p[7].contents[0]) 
len(p[7].contents)
p[7].contents[4]


# In[34]:


last_words = []


# In[39]:


#last_words1 = []


# In[41]:


#last_words2 =[]


# In[59]:


last1 = []


# In[60]:


string1 = ['Last Statement:',' Last Statement:',' Last Statement:  '
                                              ,'Last Statement:  ','Last Statement: ']

failed_links = []
for i in state_links:
    try:
        url_state = urllib.request.urlopen( 'http://www.tdcj.state.tx.us/death_row/' + i )
        doc_state = url_state.read()
        tree_state = BeautifulSoup( doc_state,'html.parser' )
        body_state = tree_state.find('div',{'id':'body'})
        p = body_state.find_all('p')
        s = ""
        for i in range(5,len(p)):
            for j in range(len(p[i].contents)):
                if type(p[i].contents[j]) != Tag and p[i].contents[j] not in string1 and p[i].contents[j] !='<br/>':
                    s = s + p[i].contents[j]
        last1.append(s)
#         time.sleep(2)
    except:
        failed_links.append(i)
        print("{} failed".format(i))
last1


# In[57]:


failed_links


# In[61]:


#last_words = last_words [:543]
len(last1)


# In[46]:


dic = {'Last Name':last_name,
      'First Name':first_name,
      'Age':age,
      'Date':date,
      'Race':race,
      'County':county,
      'Last Words':last1}
finalData = DataFrame(dic)
finalData.to_csv('C:/Users/Jabari/Desktop/NC State Analytics/Fall/Text Mining/TextMining.csv')

