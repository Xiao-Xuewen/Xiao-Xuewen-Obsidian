# 一、下载
官网直接下载：[windows](https://git-scm.com/install/windows)
在VS中检查有没有成功下载：点击左侧侧边栏的源代码管理即可
# 二、初次运行前的配置
在终端中测试
1. `git --version` 检查版本号
2. `git config --global user.name "Xiao-Xuewen"`
3. `git config --global user.email 1427602986@qq.com`
## 三、first commit
1. 进入项目目录后，`git init` 初始化本地git仓库 执行之后再执行`git status` 出现 On branch main即为正常
2. 暂存区：修改的代码可以通过`git add`进入暂存区`git commit`实际记录的是暂存区的修改,可以选择性的`git add`。
3. 进入暂存区的文件再git commit就会保存记录在本地仓库，使用`git commit -m "message"`
## 四、first push
### 4.1 配置SSH Key
1. 配置github SSH Key:终端输入：`ssh-keygen -t ed25519 -C "github邮箱"`后回车使用默认位置即可，生成之后可以在默认路径找到私钥：`id_ed25519`和公钥：`id_ed25519.pub`,**`id_ed25519` 私钥绝不能上传 GitHub、发给别人或提交进仓库**
2. 使用`Get-Content $HOME\.ssh\id_ed25519.pub`查看公钥或者直接打开公钥文件复制公钥
3. 进入github-->settings-->SSH and GPG keys-->New SSH key
4. 设置Title:自己定，例如 My Windows PC | Key type: Authentication Key | Key:`id_ed25519.pub`的完整内容
5. 测试：终端输入`ssh -T git@github.com`-->yes-->`Hi 用户名! You've successfully authenticated,`代表SSH配置成功
### 4.2 建立本地仓库与github的联系
1. 新建github仓库，右上角+-->New repository 填写仓库名 public/private
2. 我们在本地已经git init之后 下面的三个选项都不选
3. 创建完成后点击SSH 显示仓库地址
### 4.3 正式push
1. 终端执行`git remote add origin SSH地址` ，其中remote=远程仓库 add=添加 origin=这个远程仓库的本地简称（约定俗成的名字）
2. 终端执行`git remote -v`检查remote是否成功
3. 第一次push,执行`git push -u origin main`，之后执行只需`git push`
# 五、分支
1. **分支**是什么，main是默认的git主分支，实际上在底层上我们每一次commit之后git是使用指针来指向当前的状态的，也可以说是当前的版本，我们在main这个主分支上不算的增加代码，不断的add commit不断完整项目，但是如果出现一个情况，我们在项目中突然有了一个新想法，这个想法要对项目做另一个方向的修改（改方法论，改参数，增减文件），我们就可以开一个新分支branch，这个branch在底层上也是指针指向，刚刚创建时branch和main都指向同一个项目版本，不过我们在branch中修改工作区，add commit之后 branch就和main指向不同的版本了，于此同时main是不变的，我们通过switch就可以切换两个项目版本，大白话理解就是开一个新分支，就是复制当前分支的所有文件为一个新文件夹，然后在这个新文件夹上折腾修改，不影响稳定的旧文件夹且可以通过switch切换新旧文件夹（两个分支），但底层逻辑并不是“复制文件夹”
2. 分支的**合并**，我们可以通过`git merge 文件名`来合并两个分支，注意，我们执行这个命令所在的分支是接收修改的，就比如我们的branch很好 能用，那我们要merge的话 要切换回main先 然后在main中执行`git merge branch`,就可以把branch的修改合并到main中
3. 分支合并的三种情况：
   - 两个分支中的不同文件都修改了-->合并之后两个文件的修改都保留，比如main改了readme,branch改了model,在main中合并后两个文件的修改都保存了
   - 修改一个文件的不同位置-->合并后两个修改也都保留，比如main改了学习率，branch改了batch_size,合并后两个都是修改后的参数
   - 修改了同一个文件的同一个位置-->两个分支都修改了学习率且不同，这就会冲突 人工决定怎么选哪个


