// creating a scroll button to get back to the banner page

// ideally, I would like to create some sort of duration or transition time, but this has been difficult to implement


// for main page

const scrollBannerButton = document.getElementById("scroll-banner-button");

scrollBannerButton.addEventListener("click", function() {
  // Smoothly scroll the page to the banner section when the button is clicked
  window.scroll({
    behavior: "smooth",
    left: 0,
    top: document.getElementById("banner").offsetTop
  });
});

// creating a scroll button to get to the About Me page
// making it so that button doesn't work with mobile

const scrollAboutMeButton = document.getElementById("scroll-aboutme-button");

scrollAboutMeButton.addEventListener("click", function() {
  // Smoothly scroll the page to the About Me section when the button is clicked
  window.scroll({
    behavior: "smooth",
    left: 0,
    top: document.getElementById("about_me_img").offsetTop
  });
});


// for Projects page

// just setting the button as a page refresh for now. 

// const scrollProjectsButton = document.getElementById("scroll-projects-button");

// scrollProjectsButton.addEventListener("click", function() {
//   window.scrollTo({
//     behavior: "smooth",
//     top: 0
//   });
// });

