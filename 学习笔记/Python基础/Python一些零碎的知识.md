## 一、函数
- 函数就是把一些常用的代码**封装**起来，需要时直接调用 名字 参数 返回值
```
def f()

return 返回值
```
## 二、类
- 类可以理解是把**数据**和**操作这些数据的方法**放在一起
```
class Student:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print("Hello, I am", self.name)

student = Student("Tom")
student.say_hello()

```
## 三、文件I/O
## 四、"程序入口“
## 五、列表表达式

