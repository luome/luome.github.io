document.addEventListener('DOMContentLoaded', function() {
    // 语言选择器交互
    const langDropdown = document.querySelector('.lang-dropdown');
    const langToggle = document.querySelector('.lang-toggle');
    const langMenu = document.querySelector('.lang-menu');
    
    if (langToggle && langMenu) {
        langToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            langDropdown.classList.toggle('active');
        });
        
        // 点击外部关闭
        document.addEventListener('click', function(e) {
            if (!langDropdown.contains(e.target)) {
                langDropdown.classList.remove('active');
            }
        });
        
        // 更新当前语言显示
        function updateLangDisplay() {
            const currentLang = localStorage.getItem('language') || 'zh';
            const langNames = {
                'zh': '中文',
                'en': 'English',
                'ja': '日本語'
            };
            const langCurrent = document.querySelector('.lang-current');
            if (langCurrent) {
                langCurrent.textContent = langNames[currentLang] || '中文';
            }
        }
        
        // 监听语言变化
        window.addEventListener('storage', updateLangDisplay);
        updateLangDisplay();
        
        // 语言切换后更新显示
        const langButtons = document.querySelectorAll('.lang-btn');
        langButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                setTimeout(updateLangDisplay, 100);
            });
        });
    }
    
    // 平滑滚动功能
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if(targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if(targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });

    // 下拉菜单交互增强
    const dropdown = document.querySelector('.dropdown');
    const dropdownToggle = document.querySelector('.dropdown-toggle');
    const dropdownMenu = document.querySelector('.dropdown-menu');
    
    if (dropdown && dropdownToggle && dropdownMenu) {
        // 移动端点击切换下拉菜单
        dropdownToggle.addEventListener('click', function(e) {
            e.preventDefault();
            
            if (window.innerWidth <= 768) {
                const isVisible = dropdownMenu.style.display === 'block';
                dropdownMenu.style.display = isVisible ? 'none' : 'block';
            }
        });
        
        // 点击外部关闭下拉菜单
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 768 && !dropdown.contains(e.target)) {
                dropdownMenu.style.display = 'none';
            }
        });
        
        // 窗口大小改变时重置下拉菜单状态
        window.addEventListener('resize', function() {
            if (window.innerWidth > 768) {
                dropdownMenu.style.display = '';
            } else {
                dropdownMenu.style.display = 'none';
            }
        });
    }

    // 添加滚动时的渐入效果
    const fadeInElements = document.querySelectorAll('.feature-card, .screenshot');
    
    function checkFade() {
        fadeInElements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                element.classList.add('visible');
            }
        });
    }
    
    // 在加载时添加可见性类
    fadeInElements.forEach(element => {
        element.classList.add('fade-in');
    });
    
    // 监听滚动事件
    window.addEventListener('scroll', checkFade);
    
    // 初始检查
    checkFade();
}); 