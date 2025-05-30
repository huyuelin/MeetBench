window.addEventListener('scroll', () => {
  if (window.scrollY > 60) {
    document.body.classList.add('scrolled');
  } else {
    document.body.classList.remove('scrolled');
  }
}); 
