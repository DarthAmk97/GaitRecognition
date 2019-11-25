from tkinter import *

root = Tk()  # create root window
root.title("Basic GUI Layout")  # title of the GUI window
root.maxsize(900, 600)  # specify the max size the window can expand to
root.config(bg="skyblue")  # specify background color

# Create left and right frames
left_frame = Frame(root, width=200, height=400, bg='grey')
left_frame.grid(row=0, column=0, padx=10, pady=5)
right_frame = Frame(root, width=650, height=400, bg='grey')
right_frame.grid(row=0, column=1, padx=10, pady=5)

# Create frames and labels in left_frame
Label(left_frame, text="Original Image").grid(row=0, column=0, padx=5, pady=5)

# load image to be "edited"
image = PhotoImage(file="person-man.png")
original_image = image.subsample(3, 3)  # resize image using subsample
Label(left_frame, image=original_image).grid(row=1, column=0, padx=5, pady=5)

# Display image in right_frame
Label(right_frame, image=image).grid(row=0, column=0, padx=5, pady=5)

# Create tool bar frame
tool_bar = Frame(left_frame, width=250, height=200)
tool_bar.grid(row=2, column=0, padx=5, pady=5)

Button(tool_bar, text="Save", width= 20).grid(row=1, column=0, padx=5, pady=5)
Button(tool_bar, text="Delete",width= 20).grid(row=2, column=0, padx=5, pady=5)


list1 = ['live record','Verify person'];
c=StringVar()
droplist=OptionMenu(tool_bar,c, *list1)
droplist.config(width=18)
c.set('load')
droplist.grid(row = 3, column=0)
Button(tool_bar, text="Recognise",width= 20).grid(row=4, column=0, padx=5, pady=5)
Button(tool_bar, text="Delete",width= 20).grid(row=5, column=0, padx=5, pady=5)

Label(tool_bar, text="Current Database object",width= 20).grid(row=6, column=0, padx=5, pady=5)
Label(tool_bar, text="recognised objects names").grid(row=7, column=0, padx=5, pady=5)


root.mainloop()
