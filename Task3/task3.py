# TASK #3 (FOR ALL GROUPS; submit the solution to the bot)

# Use the language of your specialization (Python or PHP).

# Implement a web method accessible via HTTP GET that accepts two natural numbers, x and y, and returns their lowest common multiple. The result should be a plain string containing only digits — not HTML or any other format. If either x or y is not a non-negative integer, return the string NaN.

# Deploy the method so that it is accessible on the Internet at a URL that ends with your email address, where all characters except English letters and digits are replaced with underscores. For example, if your email is p.lebedev@itransition.com, the URL might be: http://something.something.com:port/something/p_lebedev_itransition_com. 

# Append query parameters ?x={}&y={} to the end of the URL during solution submission. To submit your solution, use the following template:
# !task3 md.smith2@mail-srv.com http://free.hoster.org/app/md_smith2_mail_srv_com?x={}&y={} 

# It’s highly recommended to wake up your server before each solution submission.


from flask import Flask,request
import math
import os

app = Flask(__name__)

@app.route('/<path:email_path>')

def calculate_lcm(email_path):
    try:
        x = request.args.get('x')
        y = request.args.get('y')

        if x is None or y is None:
            return 'NaN', 200, {'Content-Type': 'text/plain'}
        
        if '.' in str(x) or '.' in str(y):
            return 'NaN', 200, {'Content-Type': 'text/plain'}
        
        x_int = int(x)
        y_int = int(y)
        
        if x_int < 0 or y_int < 0:
            return 'NaN', 200, {'Content-Type': 'text/plain'}
        
        result = math.lcm(x_int, y_int)
        
        return str(result), 200, {'Content-Type': 'text/plain'}
        
    except (ValueError, TypeError):
        return 'NaN', 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)