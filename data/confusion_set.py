"""
混淆集模块
支持形近字、音近字的混淆替换
"""

import json
import os
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import random
from collections import defaultdict


@dataclass
class ConfusionEntry:
    """混淆项"""
    char: str                           # 原字符
    confusions: List[str]               # 混淆字符列表
    confusion_type: str = "unknown"     # 混淆类型：shape(形近), pinyin(音近), stroke(笔画近)
    weights: Optional[List[float]] = None  # 各混淆字符的权重


class ConfusionSet:
    """
    混淆集管理类
    
    支持：
    1. 加载默认混淆集（内置形近字、音近字）
    2. 加载自定义混淆集文件
    3. 合并多个混淆集
    4. 按类型筛选混淆字符
    """
    
    # 默认内置混淆集（常见形近字）
    DEFAULT_SHAPE_CONFUSIONS = {
        # 常见形近字对
        '的': ['得', '地'],
        '得': ['的', '地'],
        '地': ['的', '得'],
        '在': ['再', '载'],
        '再': ['在'],
        '做': ['作', '座'],
        '作': ['做', '坐'],
        '已': ['己', '以'],
        '己': ['已', '记'],
        '以': ['已', '亿'],
        '哪': ['那'],
        '那': ['哪', '拿'],
        '拿': ['那', '挪'],
        '像': ['象', '相'],
        '象': ['像', '相'],
        '相': ['像', '象', '想'],
        '想': ['相', '响'],
        '因': ['困', '因'],
        '困': ['因', '围'],
        '围': ['困', '韦'],
        '木': ['本', '末'],
        '本': ['木', '末'],
        '末': ['木', '本'],
        '未': ['末', '来'],
        '末': ['未', '来'],
        '日': ['目', '曰'],
        '目': ['日', '自'],
        '曰': ['日', '田'],
        '田': ['由', '曰', '甲', '申'],
        '由': ['田', '甲', '申', '电'],
        '甲': ['田', '由', '申'],
        '申': ['田', '由', '甲', '电'],
        '电': ['申', '由'],
        '买': ['卖', '实'],
        '卖': ['买'],
        '大': ['太', '天', '夫', '夭'],
        '太': ['大', '天'],
        '天': ['大', '太', '夫'],
        '夫': ['大', '天', '矢'],
        '人': ['入', '八', '个'],
        '入': ['人', '八'],
        '八': ['人', '入'],
        '士': ['土', '干', '工'],
        '土': ['士', '干', '工'],
        '干': ['士', '土', '千'],
        '千': ['干', '十'],
        '十': ['千', '什'],
        '王': ['玉', '主', '丰'],
        '玉': ['王', '主'],
        '主': ['王', '玉', '住'],
        '住': ['主', '往', '柱'],
        '往': ['住', '网'],
        '网': ['往', '冈'],
        '力': ['刀', '九', '方'],
        '刀': ['力', '几'],
        '九': ['力', '几'],
        '几': ['刀', '九', '凡'],
        '凡': ['几', '风'],
        '风': ['凡', '夙'],
        '口': ['日', '曰', '囗'],
        '月': ['用', '甩', '肉'],
        '用': ['月', '甩', '同'],
        '同': ['用', '同', '铜'],
        '心': ['必', '忄'],
        '必': ['心', '毕'],
        '手': ['毛', '手'],
        '毛': ['手', '尾'],
        '长': ['厂', '长'],
        '厂': ['长', '广'],
        '广': ['厂', '严'],
        '言': ['信', '语'],
        '信': ['言', '倍'],
        '语': ['言', '话'],
        '话': ['语', '说'],
        '说': ['话', '脱'],
        '今': ['令', '金', '禽'],
        '令': ['今', '领', '岭'],
        '金': ['今', '全'],
        '全': ['金', '余'],
        '余': ['全', '徐'],
        '气': ['汽', '氧', '氢'],
        '汽': ['气', '氧'],
        '白': ['百', '自', '目'],
        '百': ['白', '伯'],
        '自': ['白', '目', '鼻'],
        '见': ['贝', '现', '见'],
        '贝': ['见', '见'],
        '现': ['见', '观'],
        '观': ['现', '见'],
        '青': ['清', '情', '晴', '请', '精'],
        '清': ['青', '情', '晴', '请', '精'],
        '情': ['青', '清', '晴', '请', '精'],
        '晴': ['青', '清', '情', '请', '精'],
        '请': ['青', '清', '情', '晴', '精'],
        '精': ['青', '清', '情', '晴', '请'],
        '工': ['公', '功', '攻', '贡'],
        '公': ['工', '功', '攻', '贡'],
        '功': ['工', '公', '攻', '贡'],
        '攻': ['工', '公', '功', '贡'],
        '贡': ['工', '公', '功', '攻'],
        '分': ['份', '芬', '纷', '坟'],
        '份': ['分', '芬', '纷'],
        '芬': ['分', '份', '纷'],
        '纷': ['分', '份', '芬'],
        '包': ['抱', '泡', '炮', '胞', '饱'],
        '抱': ['包', '泡', '炮', '胞', '饱'],
        '泡': ['包', '抱', '炮', '胞', '饱'],
        '炮': ['包', '抱', '泡', '胞', '饱'],
        '胞': ['包', '抱', '泡', '炮', '饱'],
        '饱': ['包', '抱', '泡', '炮', '胞'],
    }
    
    # 默认内置混淆集（常见音近字）
    DEFAULT_PINYIN_CONFUSIONS = {
        # 常见同音字
        '他': ['她', '它', '塔', '踏'],
        '她': ['他', '它'],
        '它': ['他', '她'],
        '的': ['得', '地', '德'],
        '得': ['的', '地', '德'],
        '地': ['的', '得', '弟'],
        '是': ['事', '实', '时', '始', '使', '式', '室'],
        '事': ['是', '实', '时', '世', '市', '势', '示'],
        '时': ['是', '事', '实', '十', '石', '识', '食'],
        '他': ['她', '它', '塔'],
        '也': ['业', '夜', '页', '液', '野'],
        '业': ['也', '夜', '页', '液', '野'],
        '这': ['哲', '着', '遮'],
        '着': ['这', '找', '招', '照', '朝'],
        '一': ['已', '以', '义', '亿', '艺', '忆', '译', '益'],
        '以': ['已', '一', '义', '亿', '艺', '忆', '译', '益'],
        '已': ['以', '一', '义', '亿', '艺', '忆', '译', '益'],
        '有': ['又', '右', '友', '幼', '油', '由', '游', '优'],
        '又': ['有', '右', '友', '幼', '油', '由', '游', '优'],
        '和': ['合', '河', '何', '喝', '贺', '核', '禾'],
        '合': ['和', '河', '何', '喝', '贺', '核', '禾'],
        '不': ['布', '部', '步'],
        '布': ['不', '部', '步', '捕', '补'],
        '部': ['不', '布', '步'],
        '对': ['队', '兑', '堆'],
        '队': ['对', '兑', '堆', '推'],
        '做': ['作', '座', '坐', '左'],
        '作': ['做', '座', '坐', '左'],
        '座': ['做', '作', '坐', '左'],
        '坐': ['做', '作', '座', '左'],
        '那': ['哪', '拿', '呐', '纳', '钠'],
        '哪': ['那', '拿', '呐', '纳', '钠'],
        '拿': ['那', '哪', '呐', '纳', '钠'],
        '个': ['各', '格', '歌', '割', '阁', '隔'],
        '各': ['个', '格', '歌', '割', '阁', '隔'],
        '过': ['国', '果', '锅', '郭'],
        '国': ['过', '果', '锅', '郭'],
        '果': ['过', '国', '锅', '郭'],
        '没': ['每', '美', '妹', '梅', '媒', '煤', '霉'],
        '每': ['没', '美', '妹', '梅', '媒', '煤', '霉'],
        '美': ['没', '每', '妹', '梅', '媒', '煤', '霉'],
        '里': ['理', '礼', '力', '立', '利', '离', '历'],
        '理': ['里', '礼', '力', '立', '利', '离', '历'],
        '力': ['里', '理', '礼', '立', '利', '离', '历'],
        '看': ['刊', '砍', '坎', '堪'],
        '很': ['恨', '痕', '狠'],
        '恨': ['很', '痕', '狠'],
        '让': ['嚷', '壤', '瓤'],
        '想': ['响', '向', '像', '象', '项', '巷', '相', '香', '箱', '乡'],
        '向': ['想', '响', '像', '象', '项', '巷', '相', '香', '箱', '乡'],
        '像': ['想', '响', '向', '象', '项', '巷', '相', '香', '箱', '乡'],
        '象': ['想', '响', '向', '像', '项', '巷', '相', '香', '箱', '乡'],
        '相': ['想', '响', '向', '像', '象', '项', '巷', '香', '箱', '乡'],
        '给': ['级', '即', '集', '及', '己', '计', '记', '几', '技', '纪'],
        '及': ['给', '级', '即', '集', '己', '计', '记', '几', '技', '纪'],
        '既': ['济', '继', '际', '记', '季', '技', '迹'],
        '成': ['城', '程', '乘', '称', '诚', '承', '呈', '撑'],
        '城': ['成', '程', '乘', '称', '诚', '承', '呈', '撑'],
        '程': ['成', '城', '乘', '称', '诚', '承', '呈', '撑'],
        '而': ['二', '尔', '耳', '儿'],
        '二': ['而', '尔', '耳', '儿'],
        '其': ['期', '齐', '旗', '棋', '奇', '骑', '起', '器', '气', '汽'],
        '期': ['其', '齐', '旗', '棋', '奇', '骑', '起', '器', '气', '汽'],
        '齐': ['其', '期', '旗', '棋', '奇', '骑', '起', '器', '气', '汽'],
        '所': ['索', '锁', '缩'],
        '被': ['备', '背', '杯', '悲', '倍', '辈'],
        '备': ['被', '背', '杯', '悲', '倍', '辈'],
        '背': ['被', '备', '杯', '悲', '倍', '辈'],
    }
    
    def __init__(
        self,
        use_default_shape: bool = True,
        use_default_pinyin: bool = True,
        custom_confusion_files: Optional[List[str]] = None,
    ):
        """
        初始化混淆集
        
        Args:
            use_default_shape: 是否使用默认形近字混淆集
            use_default_pinyin: 是否使用默认音近字混淆集
            custom_confusion_files: 自定义混淆集文件路径列表
        """
        # 混淆字典: char -> List[Tuple[confusion_char, type, weight]]
        self.confusion_dict: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        
        # 加载默认混淆集
        if use_default_shape:
            self._load_default_confusions(self.DEFAULT_SHAPE_CONFUSIONS, "shape")
        
        if use_default_pinyin:
            self._load_default_confusions(self.DEFAULT_PINYIN_CONFUSIONS, "pinyin")
        
        # 加载自定义混淆集
        if custom_confusion_files:
            for file_path in custom_confusion_files:
                self.load_from_file(file_path)
    
    def _load_default_confusions(
        self,
        confusions: Dict[str, List[str]],
        confusion_type: str
    ):
        """加载默认混淆集"""
        for char, conf_chars in confusions.items():
            for conf_char in conf_chars:
                self.confusion_dict[char].append((conf_char, confusion_type, 1.0))
    
    def load_from_file(self, file_path: str) -> int:
        """
        从文件加载混淆集
        
        支持格式：
        1. JSON 格式: {"char": ["conf1", "conf2"], ...}
        2. TSV 格式: char\tconf1\tconf2...
        3. JSON Lines 格式: {"char": "x", "confusions": ["a", "b"], "type": "shape"}
        
        Returns:
            加载的条目数量
        """
        if not os.path.exists(file_path):
            print(f"Warning: Confusion set file not found: {file_path}")
            return 0
        
        count = 0
        ext = os.path.splitext(file_path)[1].lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                data = json.load(f)
                if isinstance(data, dict):
                    for char, confusions in data.items():
                        for conf_char in confusions:
                            self.confusion_dict[char].append((conf_char, "custom", 1.0))
                            count += 1
                elif isinstance(data, list):
                    for item in data:
                        char = item.get('char', '')
                        confusions = item.get('confusions', [])
                        conf_type = item.get('type', 'custom')
                        for conf_char in confusions:
                            self.confusion_dict[char].append((conf_char, conf_type, 1.0))
                            count += 1
            else:
                # TSV 或纯文本格式
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        char = parts[0]
                        for conf_char in parts[1:]:
                            if conf_char:
                                self.confusion_dict[char].append((conf_char, "custom", 1.0))
                                count += 1
        
        print(f"Loaded {count} confusion entries from {file_path}")
        return count
    
    def get_confusions(
        self,
        char: str,
        confusion_type: Optional[str] = None,
        max_num: Optional[int] = None
    ) -> List[str]:
        """
        获取字符的混淆字符列表
        
        Args:
            char: 原字符
            confusion_type: 混淆类型筛选（shape/pinyin/custom/None表示全部）
            max_num: 最大返回数量
            
        Returns:
            混淆字符列表
        """
        if char not in self.confusion_dict:
            return []
        
        confusions = self.confusion_dict[char]
        
        # 按类型筛选
        if confusion_type:
            confusions = [(c, t, w) for c, t, w in confusions if t == confusion_type]
        
        # 去重
        unique_chars = list(dict.fromkeys([c for c, t, w in confusions]))
        
        # 限制数量
        if max_num:
            unique_chars = unique_chars[:max_num]
        
        return unique_chars
    
    def get_random_confusion(
        self,
        char: str,
        confusion_type: Optional[str] = None
    ) -> Optional[str]:
        """
        获取一个随机的混淆字符
        
        Args:
            char: 原字符
            confusion_type: 混淆类型筛选
            
        Returns:
            随机混淆字符，如果没有则返回 None
        """
        confusions = self.get_confusions(char, confusion_type)
        if confusions:
            return random.choice(confusions)
        return None
    
    def has_confusion(self, char: str) -> bool:
        """检查字符是否有混淆集"""
        return char in self.confusion_dict and len(self.confusion_dict[char]) > 0
    
    def add_confusion(
        self,
        char: str,
        confusion_char: str,
        confusion_type: str = "custom",
        weight: float = 1.0
    ):
        """添加一个混淆对"""
        self.confusion_dict[char].append((confusion_char, confusion_type, weight))
    
    def get_all_chars(self) -> Set[str]:
        """获取所有有混淆集的字符"""
        return set(self.confusion_dict.keys())
    
    def __len__(self) -> int:
        """返回混淆集中的字符数量"""
        return len(self.confusion_dict)
    
    def __contains__(self, char: str) -> bool:
        """检查字符是否在混淆集中"""
        return char in self.confusion_dict
    
    def to_dict(self) -> Dict[str, List[str]]:
        """转换为简单字典格式（用于保存）"""
        result = {}
        for char, confusions in self.confusion_dict.items():
            result[char] = list(dict.fromkeys([c for c, t, w in confusions]))
        return result
    
    def save(self, file_path: str):
        """保存混淆集到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Confusion set saved to {file_path}")
    
    def stats(self) -> Dict[str, int]:
        """获取混淆集统计信息"""
        type_counts = defaultdict(int)
        total_pairs = 0
        
        for char, confusions in self.confusion_dict.items():
            for conf_char, conf_type, weight in confusions:
                type_counts[conf_type] += 1
                total_pairs += 1
        
        return {
            'num_chars': len(self.confusion_dict),
            'total_pairs': total_pairs,
            'type_counts': dict(type_counts)
        }


# 便捷函数：创建默认混淆集
def create_default_confusion_set() -> ConfusionSet:
    """创建默认混淆集（包含形近字和音近字）"""
    return ConfusionSet(use_default_shape=True, use_default_pinyin=True)


def create_shape_only_confusion_set() -> ConfusionSet:
    """创建仅包含形近字的混淆集"""
    return ConfusionSet(use_default_shape=True, use_default_pinyin=False)


def create_pinyin_only_confusion_set() -> ConfusionSet:
    """创建仅包含音近字的混淆集"""
    return ConfusionSet(use_default_shape=False, use_default_pinyin=True)


def load_confusion_set(file_path: str) -> ConfusionSet:
    """从文件加载混淆集"""
    confusion_set = ConfusionSet(use_default_shape=False, use_default_pinyin=False)
    confusion_set.load_from_file(file_path)
    return confusion_set
