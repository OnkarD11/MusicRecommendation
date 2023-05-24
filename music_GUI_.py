

import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
#import video_capture as value
#import lecture_details as detail_data
#import video_second as video1

#import lecture_video  as video

global fn
fn = ""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="brown")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Music Recommendation  System")

# 43

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
#image2 = Image.open('bg5.jpg')
#image2 = image2.resize((1600, 900), Image.ANTIALIAS)

#background_image = ImageTk.PhotoImage(image2)

#background_label = tk.Label(root, image=background_image)

#background_label.image = background_image

#background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
#
# function to change to next image
img=ImageTk.PhotoImage(Image.open("bg1.png"))

img2=ImageTk.PhotoImage(Image.open("bg4.jpg"))

img3=ImageTk.PhotoImage(Image.open("bg3.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=0)

x = 1
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img)
	elif x == 2:
		logo_label.config(image=img2)
	elif x == 3:
		logo_label.config(image=img3)
	x = x+1
	root.after(2000, move)

# calling the function
move() #, relwidth=1, relheight=1)

label_l1 = tk.Label(root, text="Music Recommendation System",font=("bookman old style", 35, 'bold'),
                    background="purple", fg="white", width=30, height=1)
label_l1.place(x=380, y=10)





#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#def clear_img():
#    img11 = tk.Label(root, background='bisque2')
#    img11.place(x=0, y=0)


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def cap_video():
    
#     video1.upload()
#     #from subprocess import call
#     #call(['python','video_second.py'])

def reg():
    from subprocess import call
    call(["python","music_registration.py"])

def log():
    from subprocess import call
    call(["python","music_login.py"]) 
    
def window():
  root.destroy()


button1 = tk.Button(root, text="Login", bd=5,command=log, width=14, height=1,font=('times', 20, ' bold '), bg="lightblue", fg="black")
button1.place(x=700, y=300)
def on_enter(e):
    button1['background'] = 'green'

def on_leave(e):
    button1['background'] = 'lightblue'

button1.bind("<Enter>", on_enter)
button1.bind("<Leave>", on_leave)


button2 = tk.Button(root, text="Register",bd=5,command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="lightblue", fg="black")
button2.place(x=700, y=400)
def on_enter(e):
    button2['background'] = 'green'

def on_leave(e):
    button2['background'] = 'lightblue'

button2.bind("<Enter>", on_enter)
button2.bind("<Leave>", on_leave)

button3 = tk.Button(root, text="Exit",bd=5,command=window,width=14, height=1,font=('times', 20, ' bold '), bg="red", fg="white")
button3.place(x=700, y=500)
def on_enter(e):
    button3['background'] = 'blue'

def on_leave(e):
    button3['background'] = 'red'

button3.bind("<Enter>", on_enter)
button3.bind("<Leave>", on_leave)

root.mainloop()

