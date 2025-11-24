// 多语言数据
const translations = {
    zh: {
        // 导航
        nav: {
            apps: "我们的应用",
            silens: "Silens",
            ondoku: "OnDoku 音读",
            linellae: "Linellae",
            privacy: "隐私政策"
        },
        // Hero 区域
        hero: {
            title: "智能工具，助力学习与生活",
            subtitle: "精心打造的智能应用，让学习与信息管理更高效。",
            silensBtn: "了解 Silens",
            ondokuBtn: "了解 OnDoku 音读",
            linellaeBtn: "了解 Linellae"
        },
        // 应用概览
        appsOverview: {
            title: "我们的应用",
            subtitle: "三款精心打造的应用，满足您不同的需求"
        },
        // Silens
        silens: {
            title: "Silens",
            subtitle: "AI驱动的智能体相册工具",
            description: "一款AI驱动的智能体相册工具，可以智能地创建相册、与照片对话、分析照片。我们高度重视您的隐私，所有AI处理均在安全的环境中进行，您的照片数据不会被用于训练AI模型或分享给第三方。",
            features: {
                privacy: "智能创建相册",
                ai: "与照片对话",
                lock: "AI照片分析",
                images: "隐私优先保护"
            },
            comingSoon: "即将上线",
            privacyLink: "查看隐私声明"
        },
        // OnDoku
        ondoku: {
            title: "OnDoku 音读",
            subtitle: "专业的日语音读与词汇学习工具",
            description: "专为中文学习者打造的日语音读与词汇学习工具。通过独特的中文拼音辅助，轻松掌握日语的吴音与汉音，结合系统化的词汇学习功能，全面提升发音、记忆与考试能力。",
            features: {
                language: "中文拼音辅助学习",
                volume: "吴音与汉音发音练习",
                graduation: "N1-N5考试备考",
                book: "系统化词汇学习"
            },
            privacyLink: "查看隐私政策"
        },
        // Linellae
        linellae: {
            title: "Linellae",
            subtitle: "AI驱动的书签与资讯管理工具",
            description: "用AI超能力改变您的书签体验。将链接转化为知识，用AI驱动的洞察探索Hacker News——让您的所有信息都得到智能增强。",
            features: {
                brain: "AI智能分析和总结",
                bookmark: "智能书签管理",
                newspaper: "Hacker News深度洞察",
                search: "强大的搜索功能"
            }
        },
        // 页脚
        footer: {
            rights: "保留所有权利",
            contact: "联系我们",
            silensPrivacy: "Silens 隐私声明",
            ondokuPrivacy: "OnDoku 隐私政策",
            linellaePrivacy: "Linellae 隐私政策"
        }
    },
    en: {
        nav: {
            apps: "Our Apps",
            silens: "Silens",
            ondoku: "OnDoku",
            linellae: "Linellae",
            privacy: "Privacy Policy"
        },
        hero: {
            title: "Smart Tools for Learning and Life",
            subtitle: "Carefully crafted intelligent applications to make learning and information management more efficient.",
            silensBtn: "Learn about Silens",
            ondokuBtn: "Learn about OnDoku",
            linellaeBtn: "Learn about Linellae"
        },
        appsOverview: {
            title: "Our Apps",
            subtitle: "Three carefully crafted applications to meet your different needs"
        },
        silens: {
            title: "Silens",
            subtitle: "AI-powered intelligent photo album agent",
            description: "An AI-powered intelligent photo album agent that can intelligently create albums, chat with photos, and analyze photos. We highly value your privacy. All AI processing is conducted in a secure environment, and your photo data will not be used to train AI models or shared with third parties.",
            features: {
                privacy: "Intelligent album creation",
                ai: "Chat with photos",
                lock: "AI photo analysis",
                images: "Privacy-first protection"
            },
            comingSoon: "Coming Soon",
            privacyLink: "View Privacy Statement"
        },
        ondoku: {
            title: "OnDoku",
            subtitle: "Professional Japanese pronunciation and vocabulary learning tool",
            description: "A Japanese pronunciation and vocabulary learning tool designed for Chinese learners. Through unique Chinese pinyin assistance, easily master Japanese Go-on and Kan-on pronunciations, combined with systematic vocabulary learning features to comprehensively improve pronunciation, memory, and exam capabilities.",
            features: {
                language: "Chinese pinyin-assisted learning",
                volume: "Go-on and Kan-on pronunciation practice",
                graduation: "N1-N5 exam preparation",
                book: "Systematic vocabulary learning"
            },
            privacyLink: "View Privacy Policy"
        },
        linellae: {
            title: "Linellae",
            subtitle: "AI-powered bookmark and information management tool",
            description: "Transform your bookmark experience with AI superpowers. Turn links into knowledge, explore Hacker News with AI-driven insights—make all your information intelligently enhanced.",
            features: {
                brain: "AI intelligent analysis and summarization",
                bookmark: "Smart bookmark management",
                newspaper: "Hacker News deep insights",
                search: "Powerful search functionality"
            }
        },
        footer: {
            rights: "All rights reserved",
            contact: "Contact Us",
            silensPrivacy: "Silens Privacy Statement",
            ondokuPrivacy: "OnDoku Privacy Policy",
            linellaePrivacy: "Linellae Privacy Policy"
        }
    },
    ja: {
        nav: {
            apps: "私たちのアプリ",
            silens: "Silens",
            ondoku: "OnDoku 音読",
            linellae: "Linellae",
            privacy: "プライバシーポリシー"
        },
        hero: {
            title: "学習と生活をサポートするスマートツール",
            subtitle: "学習と情報管理をより効率的にする、慎重に作られたインテリジェントアプリケーション。",
            silensBtn: "Silensについて",
            ondokuBtn: "OnDoku 音読について",
            linellaeBtn: "Linellaeについて"
        },
        appsOverview: {
            title: "私たちのアプリ",
            subtitle: "あなたの異なるニーズを満たす3つの慎重に作られたアプリケーション"
        },
        silens: {
            title: "Silens",
            subtitle: "AI駆動のインテリジェントフォトアルバムエージェント",
            description: "インテリジェントにアルバムを作成し、写真と対話し、写真を分析できるAI駆動のインテリジェントフォトアルバムエージェント。プライバシーを非常に重視しており、すべてのAI処理は安全な環境で行われ、写真データはAIモデルのトレーニングに使用されたり、第三者と共有されたりすることはありません。",
            features: {
                privacy: "インテリジェントなアルバム作成",
                ai: "写真との対話",
                lock: "AI写真分析",
                images: "プライバシー優先の保護"
            },
            comingSoon: "近日公開",
            privacyLink: "プライバシー声明を表示"
        },
        ondoku: {
            title: "OnDoku 音読",
            subtitle: "プロフェッショナルな日本語音読と語彙学習ツール",
            description: "中国語学習者向けに設計された日本語音読と語彙学習ツール。独特な中国語ピンイン補助により、日本語の呉音と漢音を簡単に習得し、体系的な語彙学習機能と組み合わせて、発音、記憶、試験能力を総合的に向上させます。",
            features: {
                language: "中国語ピンイン補助学習",
                volume: "呉音と漢音の発音練習",
                graduation: "N1-N5試験対策",
                book: "体系的な語彙学習"
            },
            privacyLink: "プライバシーポリシーを表示"
        },
        linellae: {
            title: "Linellae",
            subtitle: "AI駆動のブックマークと情報管理ツール",
            description: "AIの超能力でブックマーク体験を変革。リンクを知識に変換し、AI駆動の洞察でHacker Newsを探索—すべての情報をインテリジェントに強化します。",
            features: {
                brain: "AIインテリジェント分析と要約",
                bookmark: "スマートブックマーク管理",
                newspaper: "Hacker Newsの深い洞察",
                search: "強力な検索機能"
            }
        },
        footer: {
            rights: "全著作権所有",
            contact: "お問い合わせ",
            silensPrivacy: "Silens プライバシー声明",
            ondokuPrivacy: "OnDoku プライバシーポリシー",
            linellaePrivacy: "Linellae プライバシーポリシー"
        }
    }
};

// 获取当前语言
function getCurrentLanguage() {
    return localStorage.getItem('language') || 'zh';
}

// 设置语言
function setLanguage(lang) {
    localStorage.setItem('language', lang);
    document.documentElement.lang = lang;
    updatePageContent();
}

// 更新页面内容
function updatePageContent() {
    const lang = getCurrentLanguage();
    const t = translations[lang];
    
    if (!t) return;
    
    // 更新导航
    const navElements = {
        'nav-apps': t.nav.apps,
        'nav-silens': t.nav.silens,
        'nav-ondoku': t.nav.ondoku,
        'nav-linellae': t.nav.linellae,
        'nav-privacy': t.nav.privacy
    };
    
    Object.keys(navElements).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
            // 如果是下拉菜单的标题，只更新文本，保留图标
            if (key === 'nav-privacy' && element.querySelector('i')) {
                const icon = element.querySelector('i');
                element.innerHTML = navElements[key] + ' ';
                element.appendChild(icon);
            } else {
                element.textContent = navElements[key];
            }
        }
    });
    
    // 更新下拉菜单中的链接
    const navSilensPrivacy = document.getElementById('nav-silens-privacy');
    const navOndokuPrivacy = document.getElementById('nav-ondoku-privacy');
    const navLinellaePrivacy = document.getElementById('nav-linellae-privacy');
    if (navSilensPrivacy) navSilensPrivacy.textContent = t.footer.silensPrivacy;
    if (navOndokuPrivacy) navOndokuPrivacy.textContent = t.footer.ondokuPrivacy;
    if (navLinellaePrivacy) navLinellaePrivacy.textContent = t.footer.linellaePrivacy;
    
    // 更新 Hero 区域
    const heroTitle = document.getElementById('hero-title');
    const heroSubtitle = document.getElementById('hero-subtitle');
    const silensBtn = document.getElementById('hero-silens-btn');
    const ondokuBtn = document.getElementById('hero-ondoku-btn');
    const linellaeBtn = document.getElementById('hero-linellae-btn');
    
    if (heroTitle) heroTitle.textContent = t.hero.title;
    if (heroSubtitle) heroSubtitle.textContent = t.hero.subtitle;
    if (silensBtn) silensBtn.textContent = t.hero.silensBtn;
    if (ondokuBtn) ondokuBtn.textContent = t.hero.ondokuBtn;
    if (linellaeBtn) linellaeBtn.textContent = t.hero.linellaeBtn;
    
    // 更新应用概览
    const appsTitle = document.getElementById('apps-title');
    const appsSubtitle = document.getElementById('apps-subtitle');
    if (appsTitle) appsTitle.textContent = t.appsOverview.title;
    if (appsSubtitle) appsSubtitle.textContent = t.appsOverview.subtitle;
    
    // 更新 Silens
    const silensTitle = document.getElementById('silens-title');
    const silensSubtitle = document.getElementById('silens-subtitle');
    const silensDesc = document.getElementById('silens-description');
    const silensPrivacy = document.getElementById('silens-privacy');
    const silensComingSoon = document.getElementById('silens-coming-soon');
    
    if (silensTitle) silensTitle.textContent = t.silens.title;
    if (silensSubtitle) silensSubtitle.textContent = t.silens.subtitle;
    if (silensDesc) silensDesc.textContent = t.silens.description;
    if (silensPrivacy) silensPrivacy.textContent = t.silens.privacyLink;
    if (silensComingSoon) silensComingSoon.textContent = t.silens.comingSoon;
    
    // 更新 Silens 特性列表
    const silensFeatures = document.querySelectorAll('#silens-features li');
    if (silensFeatures.length >= 4) {
        silensFeatures[0].querySelector('span').textContent = t.silens.features.privacy;
        silensFeatures[1].querySelector('span').textContent = t.silens.features.ai;
        silensFeatures[2].querySelector('span').textContent = t.silens.features.lock;
        silensFeatures[3].querySelector('span').textContent = t.silens.features.images;
    }
    
    // 更新 OnDoku
    const ondokuTitle = document.getElementById('ondoku-title');
    const ondokuSubtitle = document.getElementById('ondoku-subtitle');
    const ondokuDesc = document.getElementById('ondoku-description');
    const ondokuPrivacy = document.getElementById('ondoku-privacy');
    
    if (ondokuTitle) ondokuTitle.textContent = t.ondoku.title;
    if (ondokuSubtitle) ondokuSubtitle.textContent = t.ondoku.subtitle;
    if (ondokuDesc) ondokuDesc.textContent = t.ondoku.description;
    if (ondokuPrivacy) ondokuPrivacy.textContent = t.ondoku.privacyLink;
    
    // 更新 OnDoku 特性列表
    const ondokuFeatures = document.querySelectorAll('#ondoku-features li');
    if (ondokuFeatures.length >= 4) {
        ondokuFeatures[0].querySelector('span').textContent = t.ondoku.features.language;
        ondokuFeatures[1].querySelector('span').textContent = t.ondoku.features.volume;
        ondokuFeatures[2].querySelector('span').textContent = t.ondoku.features.graduation;
        ondokuFeatures[3].querySelector('span').textContent = t.ondoku.features.book;
    }
    
    // 更新 Linellae
    const linellaeTitle = document.getElementById('linellae-title');
    const linellaeSubtitle = document.getElementById('linellae-subtitle');
    const linellaeDesc = document.getElementById('linellae-description');
    
    if (linellaeTitle) linellaeTitle.textContent = t.linellae.title;
    if (linellaeSubtitle) linellaeSubtitle.textContent = t.linellae.subtitle;
    if (linellaeDesc) linellaeDesc.textContent = t.linellae.description;
    
    // 更新 Linellae 特性列表
    const linellaeFeatures = document.querySelectorAll('#linellae-features li');
    if (linellaeFeatures.length >= 4) {
        linellaeFeatures[0].querySelector('span').textContent = t.linellae.features.brain;
        linellaeFeatures[1].querySelector('span').textContent = t.linellae.features.bookmark;
        linellaeFeatures[2].querySelector('span').textContent = t.linellae.features.newspaper;
        linellaeFeatures[3].querySelector('span').textContent = t.linellae.features.search;
    }
    
    // 更新页脚
    const footerRights = document.getElementById('footer-rights');
    const footerContact = document.getElementById('footer-contact');
    const footerSilensPrivacy = document.getElementById('footer-silens-privacy');
    const footerOndokuPrivacy = document.getElementById('footer-ondoku-privacy');
    const footerLinellaePrivacy = document.getElementById('footer-linellae-privacy');
    
    if (footerRights) footerRights.textContent = t.footer.rights;
    if (footerContact) footerContact.textContent = t.footer.contact;
    if (footerSilensPrivacy) footerSilensPrivacy.textContent = t.footer.silensPrivacy;
    if (footerOndokuPrivacy) footerOndokuPrivacy.textContent = t.footer.ondokuPrivacy;
    if (footerLinellaePrivacy) footerLinellaePrivacy.textContent = t.footer.linellaePrivacy;
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    const lang = getCurrentLanguage();
    document.documentElement.lang = lang;
    updatePageContent();
    
    // 语言选择器事件
    const langButtons = document.querySelectorAll('.lang-btn');
    langButtons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            const selectedLang = this.getAttribute('data-lang');
            setLanguage(selectedLang);
            
            // 更新活动状态
            langButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // 设置当前语言按钮为活动状态
    const currentLangBtn = document.querySelector(`.lang-btn[data-lang="${lang}"]`);
    if (currentLangBtn) {
        currentLangBtn.classList.add('active');
    }
    
    // 更新语言选择器显示
    const langNames = {
        'zh': '中文',
        'en': 'English',
        'ja': '日本語'
    };
    const langCurrent = document.querySelector('.lang-current');
    if (langCurrent) {
        langCurrent.textContent = langNames[lang] || '中文';
    }
});

