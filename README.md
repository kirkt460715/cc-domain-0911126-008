# 托育收费与家庭负担

本项目使用 Python 3.11 及以上版本，仅依赖标准库，无需安装第三方包。数据文件均为本地合成记录，不含真实个人或机构资料。

tariffs 管理收费日历、分摊和补助；households 校验家庭、儿童及入托关系；analysis 核算家庭月支出及加权分布；publication 归档版本、复核与公开汇总。

金额使用整数分，家庭收入是全年家庭收入。押金与经营支出分开。同一家庭儿童共享补助上限；样本权重属于家庭，月份不增加家庭数。公开汇总进行小组保护，内部明细不用于公开。

在项目根目录运行完整测试：

```sh
python3 -m unittest discover -s tests -v
```

编译检查：

```sh
python3 -m compileall -q .
```

实际演示：

```sh
python3 nursery_report.py demo --fixture fixtures/county.json
```

## 家庭权重校准

校准单位是家庭，基准扩展权重（家庭行的 `weight`）永不修改；系统在原样本不变的前提下生成独立的校准权重版本。校准最小化相对基准权重的熵距离（广义 raking），在用户给定的倍率上下限内满足冻结在同一参考日期的已知边际，金额与收入值不作为调权自由变量。

支持两类约束：地区、收入状态等互斥分类边际，以及每户不同年龄段儿童个数这类计数约束。收敛阈值、迭代上限与每条约束的目标、实际值、残差都写入记录；零支撑、目标冲突、线性相关、边界内不可行会被区分报告，未满足边际时绝不会把最后一轮权重标为成功。

```sh
python3 nursery_report.py calibrate \
  --input fixtures/county_calibration.json \
  --start 2026-11-01 --end 2026-12-01 \
  --controls fixtures/population_controls.json \
  --directory /tmp/calibration-bundle
```

导出目录含 `constraints.json`（目标/基准加权/实际/残差，可逐格手算）、`household-weights.json`（每户基准权重、倍率、校准权重及是否顶到边界）、`iteration-trace.json`（每轮残差）和 `comparison.json`（按家庭汇总的基准与新权重总支出、负担分布、超负担线比例、有效样本量与极端权重）。一户跨多月只挂一个权重、金额按月累加；零收入有支出与收入缺失家庭继续单列。`fixtures/population_controls_*.json` 提供零支撑、目标冲突、容量越界与倍率阻挡四类失败样例。

权重版本通过家庭键、输入摘要、结果摘要与分类摘要绑定；多户、少户或家庭分类变化都会拒绝静默复用。归档流程可通过 `request --action calibrate` 与 `calibration-export` 子命令使用，并纳入备份/恢复完整性审计。

演示在临时目录内建立本地状态，结束后自动清理。测试、编译与演示在依赖已准备的环境内均不下载数据或软件。输入错误以非零退出码及错误对象报告。更多子命令可通过对应入口的 --help 查看。

容器环境可执行 `docker build -t local-analysis .`，再运行 `docker run --rm --network none local-analysis`。镜像固定基础内容摘要，构建后运行只访问本地文件。完整测试与演示也可直接使用已有 Python 环境。
