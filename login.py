from tkinter import *
root = Tk()
root.geometry('500x500')
root.title("Sign in")

label_0 = Label(root, text="Sign In",width=20,font=("bold", 20))
label_0.place(x=100,y=53)


label_1 = Label(root, text="Username",width=20,font=("bold", 10))
label_1.place(x=80,y=130)

entry_1 = Entry(root)
entry_1.place(x=240,y=130)

label_2 = Label(root, text="Password", width=20,font=("bold", 10))
label_2.place(x=80,y=180)

entry_2 = Entry(root)
entry_2.place(x=240,y=180)

var = IntVar()
Radiobutton(root, text="keep me logged in", variable=var, value=1).place(x=235,y=250)


Button(root, text='Log In',width=15,bg='brown',fg='white').place(x=240,y=300)


root.mainloop()