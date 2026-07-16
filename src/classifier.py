"""Keyword-based classifier — maps free-form keywords to predefined complaint types.

This is a fallback classifier used when the LLM does not return a valid
complaint_type, or for post-processing validation.
"""


# Mapping rules: each complaint_type has a list of keyword patterns.
# A match on any pattern in the list classifies to that type.
KEYWORD_RULES: dict[str, list[str]] = {
    "Course/Teaching Quality": [
        "课程质量", "讲得太浅", "太慢了", "基础", "太贵",
        "内容重复", "取消", "更新", "加密", "回放", "打不开",
        "模糊", "学费", "干货", "缩水", "错别字", "期货课程",
        "期权", "零基础", "跟不上", "太深", "听不懂",
        "报名", "承诺", "大纲", "图解", "截屏",
        "过时", "简单", "案例", "实战课", "理论课",
        "讲师", "导师", "系统课", "讲得太快",
        "课件", "作业批改", "敷衍", "名师",
        "试听", "录播", "画质",
        "讲得", "教得", "水得很", "水分", "内容不行",
        "不专业", "质量差", "浪费时间", "浪费钱",
        "没讲清楚", "没讲明白", "讲得太少", "没教",
        "拖堂", "迟到", "早退", "请假", "停课",
        "没干货", "注水", "敷衍了事", "糊弄",
        "听不懂", "跟不上", "跳过",
    ],
    "Investment Advice/Losses": [
        "亏了", "被套", "止损", "抄底", "加仓", "减仓",
        "清仓", "止盈", "推票", "喊单", "跟单", "带单",
        "推荐", "反弹", "牛市", "暴跌", "涨停",
        "收益率", "收益", "亏损", "损失", "中签",
        "内幕", "实盘", "模拟盘", "战法",
        "板块", "轮动", "预测", "大盘分析",
        "追涨", "杀跌", "杠杆", "做空",
        "策略", "波段", "操作建议", "仓位管理",
        "卖空", "平仓",
        "亏钱", "血亏", "套牢", "踏空", "卖飞",
        "回撤", "浮亏", "割肉", "接盘", "抬轿",
        "杀猪盘", "坐庄", "出货", "诱多", "诱空",
        "高抛低吸", "做T", "补仓", "半仓", "满仓",
    ],
    "Platform/App Issues": [
        "APP", "闪退", "卡顿", "数据延迟", "自选股",
        "崩溃", "加载", "同步", "登录", "推送",
        "选股器", "导出", "维护", "版本", "闪退",
        "行情", "搜索", "报警", "缓存",
        "页面", "UI", "界面", "下载", "上传",
        "排行榜", "通知", "日历", "夜间模式",
        "排版", "文件夹", "分组",
        "响应慢", "没反应", "死机", "黑屏", "白屏",
        "加载中", "转圈", "刷新", "报错", "错误",
        "打不开", "不能打开", "无法打开", "链接失效",
        "服务器", "网络", "连不上", "断线", "掉线",
        "卡死", "无响应",
    ],
    "Customer Service/Refund": [
        "退款", "客服", "人工", "排队", "投诉",
        "发票", "售后", "推诿", "态度", "拉黑",
        "电话", "邮箱", "微信", "回复",
        "补偿", "申诉", "班主任", "转接",
        "反馈", "处理", "等待",
        "态度差", "态度恶劣", "不耐烦", "骂人", "怼人",
        "不解决", "踢皮球", "敷衍", "推卸",
        "效率低", "拖", "拖延", "慢", "不及时",
        "投诉无门", "没人管", "没人理",
    ],
    "Learning Effectiveness": [
        "白学了", "听不懂", "不会选股", "没进步",
        "没学会", "没法交易", "不实用",
        "难度大", "跟不上节奏", "作业太难",
        "复利", "知识", "学习计划",
        "复习", "考试", "理论",
        "没效果", "没用", "太难", "学不会",
        "白花钱", "浪费时间", "没收获",
        "学完就忘", "记不住", "不会用",
        "基础差", "底子薄",
    ],
    "Membership/Renewal": [
        "会员", "续费", "积分", "折扣", "套餐",
        "年费", "白银", "黄金", "铂金", "钻石",
        "降级", "过期",
        "到期", "自动续费", "会员费", "会费",
        "权益", "特权", "等级", "会员价",
        "老用户", "新用户", "首充",
    ],
    "False Advertising": [
        "虚假", "夸大", "造假", "不符",
        "宣传", "马后炮", "P的", "假",
        "截图", "吹捧",
        "骗", "骗人", "骗子", "忽悠",
        "割韭菜", "坑人", "被坑", "套路",
        "虚假宣传", "货不对板", "夸张",
        "吹牛", "画大饼", "空头支票",
        "删评论", "删帖", "控评", "水军",
        "托", "假号", "小号",
    ],
    "Pricing Issues": [
        "价格", "收费", "费用", "扣费", "扣款",
        "手续费", "多收", "少退", "金额不对",
        "分期", "利息",
        "乱收费", "隐形收费", "隐藏收费",
        "涨价", "变相收费", "比外面贵",
    ],
    "Unfair Terms": [
        "霸王条款", "不退不换", "违约金", "封号",
        "条款", "合同", "转让", "绑定",
        "不讲道理", "不讲理", "无理",
        "强制", "捆绑", "霸王", "不平等",
    ],
    "Account/Order Issues": [
        "账号", "封", "登录", "密码", "解约",
        "注销", "被锁", "找回",
        "冻结", "限制", "异常", "被盗",
    ],
}


def classify_keywords(keywords: list[str]) -> str | None:
    """Map a list of free-form keywords to the best matching complaint type.

    Args:
        keywords: List of keyword strings from the LLM output.

    Returns:
        The matched complaint_type string, or None if no rule matches.

    """
    best_type: str | None = None
    best_score = 0

    for complaint_type, patterns in KEYWORD_RULES.items():
        score = 0
        for kw in keywords:
            for pattern in patterns:
                if pattern in kw:
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best_type = complaint_type

    return best_type if best_score > 0 else None


def is_valid_complaint_type(type_str: str | None) -> bool:
    """Check if a complaint_type string is one of the predefined types.

    Args:
        type_str: The complaint_type to validate.

    Returns:
        True if it matches a known type.

    """
    return type_str in KEYWORD_RULES


VALID_TYPES = list(KEYWORD_RULES.keys())
