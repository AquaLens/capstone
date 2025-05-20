// smooth scroll function
const lenis = new Lenis()

lenis.on('scroll', (e) => {
    console.log(e)
})

function raf(time) {
    lenis.raf(time)
    requestAnimationFrame(raf)
}

requestAnimationFrame(raf)

// button scroll to scene1
document.querySelector('#dive_button').addEventListener('click', function () {
    const target = document.querySelector('.scene1');
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });

// button scroll to explore
document.querySelector('#explore_button').addEventListener('click', function () {
    const target = document.querySelector('.explore');
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });

// importing gsap and ScrollTrigger
gsap.registerPlugin(ScrollTrigger);

// pin the image when its bottom hits the viewport
ScrollTrigger.create({
    trigger: ".scene1",
    start: "bottom bottom",
    end: "+=800",
    pin: true,
    anticipatePin: 1,
    scrub: true
});

// Scroll detection for header visibility
document.addEventListener("DOMContentLoaded", () => {
  const body = document.body;

  let lastScrollY = window.scrollY;

  window.addEventListener("scroll", () => {
    if (window.scrollY > lastScrollY && window.scrollY > 50) {
      // User is scrolling down
      body.classList.add("scrolled");
    } else {
      // User is scrolling up
      body.classList.remove("scrolled");
    }

    lastScrollY = window.scrollY;
  });
});

// fade in the overlay text
gsap.to(".act1_text", {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
        trigger: ".scene1",
        start: "bottom bottom",
        end: "+=600",
        scrub: true
    }
});

gsap.to(".wiggle_n", {
    y: -8,
    duration: 1.0,
    repeat: -1,
    yoyo: true,
    ease: "sine.inOut"
});
gsap.to(".wiggle_a", {
    y: -8,
    duration: 1.1,
    repeat: -1,
    yoyo: true,
    ease: "cos.inOut"
});
gsap.to(".wiggle_p", {
    y: -8,
    duration: 0.85,
    repeat: -1,
    yoyo: true,
    ease: "cos.inOut"
});

// // Create the timeline
// const tl = gsap.timeline({
//   scrollTrigger: {
//     trigger: "#plastic_bag",
//     start: "+30% top top",
//     end: "+=600", // adjust 
//     pin: ".pin_section_2",
//     scrub: true,
//     // markers: true,
//   }
// });

// Fade in the overlay text
gsap.to(".act2_text_1", {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
        trigger: "#plastic_bag",
        start: "top top",
        end: "+=600",
        scrub: true
    }
});


// ScrollTrigger.create({
//     trigger: ".pin_section_2",
//     start: "bottom bottom",
//     end: "+=800",
//     pin: true,
//     anticipatePin: 1,
//     scrub: true,
//     markers: true
// });

// Fade in the overlay text
gsap.to(".act2_text_2", {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
        trigger: ".pin_section_2",
        start: "bottom bottom",
        end: "+=400",
        scrub: true
    }
});

gsap.fromTo("#quote_1",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 0.4,
    ease: "power2.out",
    scrollTrigger: {
      trigger: ".scene3",
      start: "top top", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

gsap.fromTo("#quote_2",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 0.4,
    ease: "power2.out",
    scrollTrigger: {
      trigger: ".scene3",
      start: "15.5%+top top", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

gsap.fromTo("#quote_3",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 0.4,
    ease: "power2.out",
    scrollTrigger: {
      trigger: ".scene3",
      start: "30.5%+top top", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

gsap.fromTo("#quote_4",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 0.4,
    ease: "power2.out",
    scrollTrigger: {
      trigger: ".scene3",
      start: "46.4%+top top", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

gsap.fromTo("#quote_5",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 0.4,
    ease: "power2.out",
    scrollTrigger: {
      trigger: ".scene3",
      start: "61%+top top", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

gsap.to(".act3_text", {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
        trigger: ".scene3",
        start: "bottom bottom",
        end: "+=400",
        scrub: true
    }
});

let tl3 = gsap.timeline({
  scrollTrigger: {
    trigger: ".scene4",
    start: "top top", // when the top of ___ hits the center of viewport
    end: "+=450", // adjust 
    scrub: true,
    markers: false
  }
});

tl3.to("#hand_container", {
  x: "100%"
})

// gsap.fromTo("#project_container1",
//   { opacity: 0, y: 20 },
//   {
//     opacity: 1,
//     y: 0,
//     duration: 1,
//     ease: "power2.out",
//     scrollTrigger: {
//       trigger: "#project_container1",
//       start: "top center", // when the top of ___ hits the center of viewport
//       toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
//       markers: false  
//     }
//   }
// );

// gsap.fromTo("#project_container2",
//   { opacity: 0, y: 20 },
//   {
//     opacity: 1,
//     y: 0,
//     duration: 1,
//     ease: "power2.out",
//     scrollTrigger: {
//       trigger: "#project_container2",
//       start: "top center", // when the top of ___ hits the center of viewport
//       toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
//       markers: false 
//     }
//   }
// );

// gsap.fromTo("#project_container3",
//   { opacity: 0, y: 20 },
//   {
//     opacity: 1,
//     y: 0,
//     duration: 1,
//     ease: "power2.out",
//     scrollTrigger: {
//       trigger: "#project_container3",
//       start: "top center", // when the top of ___ hits the center of viewport
//       toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
//       markers: false 
//     }
//   }
// );

gsap.fromTo(".project_container",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
      trigger: ".project_container",
      start: "top center", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

// pin the image when its bottom hits the viewport
ScrollTrigger.create({
    trigger: ".scene5",
    start: "bottom bottom",
    end: "+=2000",
    pin: true,
    anticipatePin: 1,
    scrub: true
});

// fade in the overlay text
gsap.to("#act5_text_1", {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
        trigger: ".scene5",
        start: "bottom bottom",
        end: "+=400",
        scrub: true
    }
});

const scene5Timeline = gsap.timeline({
  scrollTrigger: {
    trigger: ".scene5",
    start: "bottom bottom",
    end: "+=800",
    scrub: true
  }
});

scene5Timeline.to("#act5_text_2", {
  opacity: 1,
  y: 0,
  duration: 2,
  ease: "power2.out",
  delay: 3 // x-second delay before this animation starts
});

// const section = document.querySelector('section.vid')
// const vid = document.querySelector('video')

// vid.pause()

// const scroll = () => {
//   const distance = window.scrollY - section.offsetTop
//   const total = section.clientHeight - window.innerHeight

//   let percentage = distance / total
//   percentage = Math.max(0, percentage)
//   percentage = Math.min(percentage, 1)

//   if (vid.duration > 0) {
//     vid.currentTime = vid.duration * percentage
//   }
// }

// scroll()
// window.addEventListener("scroll", scroll)