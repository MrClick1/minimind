# MiniMind 下载、Fork 与上游同步方案

> 适用场景：将 `jingyaogong/minimind` Fork 到自己的 GitHub 账号，在本地进行源码阅读、注释和实验，同时保留持续同步原项目更新的能力。

---

## 1. 推荐的仓库规划

建议保留两个仓库，各自承担不同职责。

### 1.1 MiniMind 源码 Fork

```text
MrClick1/minimind
```

用途：

- 保留完整 MiniMind 源码；
- 添加中文注释；
- 打印中间张量形状；
- 修改训练参数；
- 编写调试代码；
- 创建个人学习分支；
- 同步原作者后续更新。

### 1.2 MiniMind 学习记录仓库

```text
MrClick1/minimind_learn
```

用途：

- 记录源码阅读笔记；
- 编写独立的小型实验；
- 复现核心模块；
- 保存训练日志和实验结论；
- 沉淀面试总结和项目成果。

不建议将完整 MiniMind 源码直接复制进 `minimind_learn`。完整源码放在 Fork 仓库，学习成果放在学习仓库，更容易区分上游代码和个人产出。

---

## 2. 在 GitHub 上 Fork MiniMind

打开原项目：

```text
https://github.com/jingyaogong/minimind
```

点击页面右上角的 **Fork**，将其复制到自己的 GitHub 账号下。

Fork 完成后，自己的仓库地址通常为：

```text
https://github.com/MrClick1/minimind
```

Fork 的优点：

- 保留原项目完整提交历史；
- GitHub 会明确显示源码来源；
- 可以自由创建分支和修改代码；
- 后续可以方便地同步原作者更新；
- 可以将个人修改提交到自己的 GitHub；
- 有需要时可以向原项目提交 Pull Request。

---

## 3. 克隆自己的 Fork

在本地选择合适的工作目录，然后执行：

```bash
git clone https://github.com/MrClick1/minimind.git
cd minimind
```

检查当前远程仓库：

```bash
git remote -v
```

此时通常只会看到：

```text
origin  https://github.com/MrClick1/minimind.git (fetch)
origin  https://github.com/MrClick1/minimind.git (push)
```

其中：

- `origin` 指向自己的 GitHub Fork；
- 可以从 `origin` 拉取代码；
- 也可以将个人提交推送到 `origin`。

---

## 4. 添加原作者仓库为 upstream

执行：

```bash
git remote add upstream https://github.com/jingyaogong/minimind.git
```

再次检查：

```bash
git remote -v
```

预期结果：

```text
origin    https://github.com/MrClick1/minimind.git (fetch)
origin    https://github.com/MrClick1/minimind.git (push)
upstream  https://github.com/jingyaogong/minimind.git (fetch)
upstream  https://github.com/jingyaogong/minimind.git (push)
```

两个远程仓库的职责：

| 远程名称 | 指向 | 主要用途 |
|---|---|---|
| `origin` | 自己的 Fork | 保存和推送个人修改 |
| `upstream` | 原作者仓库 | 获取原项目最新更新 |

注意：虽然 `git remote -v` 可能显示 `upstream` 的 push 地址，但普通用户通常没有原作者仓库的写入权限，因此不要向 `upstream` 推送代码。

---

## 5. 建议的分支管理方式

不建议直接在主分支中添加大量学习注释和调试代码。

推荐结构：

```text
master
├── 尽量保持与原作者主分支一致
│
├── learn/model-structure
│   └── 模型结构阅读、中文注释、张量形状调试
│
├── learn/dataset
│   └── Tokenizer、PretrainDataset、SFTDataset 学习
│
├── learn/training-loop
│   └── 训练循环、梯度累积、混合精度实验
│
└── experiment/tiny-overfit
    └── 小数据过拟合实验
```

创建模型结构学习分支：

```bash
git checkout -b learn/model-structure
```

或者使用较新的 Git 命令：

```bash
git switch -c learn/model-structure
```

在学习分支中可以放心进行：

- 添加中文注释；
- 增加 `print()` 调试输出；
- 修改模型维度；
- 编写临时代码；
- 记录实验结果。

例如：

```python
print("xq shape:", xq.shape)
print("xk shape:", xk.shape)
print("xv shape:", xv.shape)
print("attention output shape:", output.shape)
```

---

## 6. 提交自己的学习修改

查看文件变化：

```bash
git status
```

添加修改：

```bash
git add .
```

提交：

```bash
git commit -m "learn: add comments for MiniMind attention"
```

推送学习分支到自己的 GitHub：

```bash
git push -u origin learn/model-structure
```

第一次使用 `-u` 后，后续通常只需要：

```bash
git push
```

推荐的提交信息示例：

```text
learn: annotate MiniMindConfig parameters
learn: trace attention tensor shapes
experiment: add tiny model forward test
experiment: verify shifted cross entropy
docs: add notes for RMSNorm and RoPE
fix: correct local dataset path
```

---

## 7. 同步原作者的最新更新

原作者更新 MiniMind 后，自己的 Fork 不一定会自动保持最新，因此需要定期同步。

### 7.1 切换到主分支

```bash
git checkout master
```

或者：

```bash
git switch master
```

确认主分支没有未提交修改：

```bash
git status
```

### 7.2 获取 upstream 最新提交

```bash
git fetch upstream
```

这条命令只会下载原作者的最新提交，不会立刻修改当前工作区。

### 7.3 合并原作者主分支

MiniMind 当前主分支通常为 `master`，执行：

```bash
git merge upstream/master
```

如果本地主分支始终没有个人修改，通常会直接进行 fast-forward，不产生额外合并提交。

也可以使用：

```bash
git pull upstream master
```

但学习阶段更推荐分开执行 `fetch` 和 `merge`，过程更清楚：

```bash
git fetch upstream
git merge upstream/master
```

### 7.4 推送到自己的 Fork

```bash
git push origin master
```

完成后的数据流：

```text
原作者仓库 upstream
        ↓ git fetch
本地 master
        ↓ git merge
本地最新代码
        ↓ git push
自己的 Fork origin
```

---

## 8. 将最新主分支合并到学习分支

主分支同步完成后，切回个人学习分支：

```bash
git checkout learn/model-structure
```

将更新后的主分支合并进来：

```bash
git merge master
```

如果没有冲突，学习分支将同时拥有：

- 原作者最新代码；
- 自己此前添加的中文注释和调试代码。

然后推送：

```bash
git push origin learn/model-structure
```

完整同步流程：

```bash
git checkout master
git fetch upstream
git merge upstream/master
git push origin master

git checkout learn/model-structure
git merge master
git push origin learn/model-structure
```

---

## 9. 更安全的同步方式：先创建临时备份

如果学习分支修改较多，合并前可以先创建一个备份分支：

```bash
git checkout learn/model-structure
git branch backup/model-structure-before-sync
```

然后再进行：

```bash
git merge master
```

即使合并过程出现问题，也可以通过备份分支找回同步前的状态。

确认没有问题后，可以删除本地备份分支：

```bash
git branch -d backup/model-structure-before-sync
```

---

## 10. 遇到未提交修改时怎么办

切换分支或同步前，如果执行：

```bash
git status
```

发现存在尚未提交的修改，建议优先提交：

```bash
git add .
git commit -m "wip: save current learning progress"
```

如果修改还不适合提交，可以暂存：

```bash
git stash push -m "temporary learning changes"
```

同步完成后恢复：

```bash
git stash pop
```

常用流程：

```bash
git stash
git checkout master
git fetch upstream
git merge upstream/master
git checkout learn/model-structure
git merge master
git stash pop
```

注意：`git stash pop` 后也可能产生冲突，需要人工处理。

---

## 11. 合并冲突处理方法

如果执行：

```bash
git merge master
```

出现冲突，Git 会在冲突文件中插入类似标记：

```text
<<<<<<< HEAD
自己的学习修改
=======
原作者的新代码
>>>>>>> master
```

需要手动决定：

- 保留自己的代码；
- 保留上游代码；
- 或将两边内容合理合并。

修改完成后：

```bash
git add 冲突文件
git commit
```

查看冲突状态：

```bash
git status
```

如果暂时不想继续合并，可以取消：

```bash
git merge --abort
```

取消后会尽量恢复到合并开始前的状态。

---

## 12. 不小心修改了 master 怎么办

如果修改还没有提交，可以先创建新分支，把当前修改保留下来：

```bash
git switch -c learn/current-work
```

这样未提交修改会跟着进入新分支。

如果已经在 `master` 提交，但还没有推送，可以：

```bash
git branch learn/current-work
git reset --hard upstream/master
```

这会：

1. 用新分支保留个人提交；
2. 将本地 `master` 恢复到上游状态。

注意：`git reset --hard` 会删除未提交修改，执行前必须先检查：

```bash
git status
```

---

## 13. 检查当前仓库状态

查看所在分支：

```bash
git branch --show-current
```

查看本地和远程分支：

```bash
git branch -a
```

查看远程仓库：

```bash
git remote -v
```

查看最近提交：

```bash
git log --oneline --graph --decorate -10
```

查看当前状态：

```bash
git status
```

查看本地 `master` 与原作者的差异：

```bash
git fetch upstream
git log master..upstream/master --oneline
```

如果没有输出，通常说明本地 `master` 已包含 upstream 的最新提交。

查看自己比上游多出的提交：

```bash
git log upstream/master..master --oneline
```

---

## 14. 更新 Fork 的推荐固定流程

以后每次更新，可以直接按照下面的顺序执行。

### 14.1 主分支同步

```bash
git switch master
git status
git fetch upstream
git merge upstream/master
git push origin master
```

### 14.2 学习分支同步

```bash
git switch learn/model-structure
git merge master
git push origin learn/model-structure
```

### 14.3 同步前存在未完成修改

```bash
git stash push -m "temporary changes before upstream sync"

git switch master
git fetch upstream
git merge upstream/master
git push origin master

git switch learn/model-structure
git merge master
git stash pop
```

---

## 15. 首次配置完整命令清单

如果已经在 GitHub 完成 Fork，可以按下面的顺序配置：

```bash
# 1. 克隆自己的 Fork
git clone https://github.com/MrClick1/minimind.git
cd minimind

# 2. 添加原作者仓库
git remote add upstream https://github.com/jingyaogong/minimind.git

# 3. 检查远程地址
git remote -v

# 4. 获取原作者最新信息
git fetch upstream

# 5. 创建学习分支
git switch -c learn/model-structure

# 6. 推送学习分支
git push -u origin learn/model-structure
```

---

## 16. `minimind_learn` 与源码 Fork 的关联

可以在 `MrClick1/minimind_learn` 的 README 中补充：

```markdown
## 相关仓库

- 上游项目：`jingyaogong/minimind`
- 我的源码学习 Fork：`MrClick1/minimind`
- 当前仓库：用于记录源码解析、独立实验、训练结果和学习笔记
```

建议在学习笔记中记录对应的源码版本，例如：

```markdown
## 对应源码版本

- Repository: `jingyaogong/minimind`
- Branch: `master`
- Commit: `89d674b8a517010f5561b6d8ab2dcbb58e2fb91b`
```

这样即使上游以后发生较大改动，也能知道笔记是基于哪个版本编写的。

查看当前提交：

```bash
git rev-parse HEAD
```

查看简短提交编号：

```bash
git rev-parse --short HEAD
```

---

## 17. 最终推荐结构

```text
GitHub
├── MrClick1/minimind
│   ├── 完整上游源码
│   ├── 学习分支
│   ├── 中文注释
│   ├── 调试代码
│   └── 训练实验
│
└── MrClick1/minimind_learn
    ├── notes
    ├── experiments
    ├── scripts
    ├── 实验结果
    └── 面试总结
```

核心原则：

1. `master` 尽量保持与 `upstream/master` 一致；
2. 个人修改放在 `learn/*` 或 `experiment/*` 分支；
3. `origin` 用于保存自己的代码；
4. `upstream` 用于获取原作者更新；
5. 完整源码和个人学习成果分开管理；
6. 每篇学习笔记记录对应的源码 commit。

---

## 18. 一句话总结

```text
Fork 用于获得一份属于自己的完整源码仓库；
origin 用于保存个人修改；
upstream 用于同步原作者更新；
学习分支用于添加注释和实验；
minimind_learn 用于沉淀真正属于自己的学习成果。
```
