#!/usr/bin/env python

# In[3]:


import tkinter as tk
import tkinter.messagebox
from recogFunc import check, capture, train, read, clean


# In[4]:


userList = read('names.csv')
status = False

root = tk.Tk()
root.title('adminitrative window')
root.geometry("600x400+200+200")

entry_name = tk.Entry(root)
entry_name.place(x=240, y=100, anchor='nw')
t = tk.Text(root,height=4,width=85)

def record():
    var = entry_name.get()
    flag = check(var, userList)
    if(flag == True):
        tkinter.messagebox.showinfo(title='Warning!', message=var+', Your face is already in the database, No need to capture!\n')
    else:
        t.insert(1.1,var+', Thank you! Capturing your face, please look the camera and wait ...\n')
        capture(userList)
        t.insert(2.1,var+', Thank you! Face captured, please click the train button ...\n')
        tkinter.messagebox.showinfo(title='Hi!', message='Record completed!')
    t.place(x=0, y=250, anchor='nw')
     
    
def gui_Train():
    t.insert(3.1,'Training...\n')
    train()    
    tkinter.messagebox.showinfo(title='Hi!', message='Training completed!')
        
r_button = tk.Button(root, text='record', width=10, height=1, command=record)
r_button.place(x=260, y=150, anchor='nw')

r_button2 = tk.Button(root, text='train', width=10, height=1, command=gui_Train)
r_button2.place(x=260, y=200, anchor='nw')
    
root.mainloop()


# In[ ]:




