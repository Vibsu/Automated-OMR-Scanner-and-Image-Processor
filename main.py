import pandas as pd
from flask import Flask, render_template, request, redirect
from usn import detect_bubbles_and_get_integrated_values
from subcode import detect_bubbles
from marksdet1 import detect_bubbles_and_get_integrated_values1
from marksdet2 import detect_bubbles_and_get_integrated_values2


app = Flask(__name__)

@app.route('/')
def index():
    imag_path = "gat10.jpg"
    usn = detect_bubbles_and_get_integrated_values(imag_path)
    subcode = detect_bubbles(imag_path)
    marks1 = detect_bubbles_and_get_integrated_values1(imag_path)
    marks2 = detect_bubbles_and_get_integrated_values2(imag_path)
    split_marks1 = marks1.split(" ")
    split_marks2 = marks2.split(" ")
    each_total = list()
    each_total.append(int(split_marks1[0]) + int(split_marks1[1]) + int(split_marks1[2]))
    each_total.append( int(split_marks1[3]) + int(split_marks1[4]) + int(split_marks1[5]))
    each_total.append( int(split_marks1[6]) + int(split_marks1[7]) + int(split_marks1[8]))
    each_total.append( int(split_marks1[9]) + int(split_marks1[10]) + int(split_marks1[11]))
    each_total.append( int(split_marks2[0]) + int(split_marks2[1]) + int(split_marks2[2]))
    each_total.append( int(split_marks2[3]) + int(split_marks2[4]) + int(split_marks2[5]))
    each_total.append( int(split_marks2[6]) + int(split_marks2[7]) + int(split_marks2[8]))
    each_total.append( int(split_marks2[9]) + int(split_marks2[10]) + int(split_marks2[11]))
    final_marks=0
    final_marks = max(each_total[0],each_total[1]) + max(each_total[2],each_total[3]) + max(each_total[4],each_total[5]) + max(each_total[6],each_total[7])
    
    return render_template('index.html', usn=usn, subcode=subcode, split_marks1=split_marks1,split_marks2=split_marks2, total = each_total, final_marks=final_marks)

if __name__ == '__main__':
    app.run(debug=False)
