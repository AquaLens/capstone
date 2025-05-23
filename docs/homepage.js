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
    const scene1 = document.querySelector('.scene1');

    if (scene1) {
        // calculate the scroll position
        const scene1Top = scene1.getBoundingClientRect().top + window.scrollY;
        const scrollTarget = scene1Top + (scene1.offsetHeight * 0.30);

        // ccroll to the calculated position
        window.scrollTo({
            top: scrollTarget,
            behavior: 'smooth'
        });
    }
});

// button scroll to explore
document.querySelector('#explore_button').addEventListener('click', function () {
  window.location.href = '/projects'; 
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

gsap.set(".wiggle_a img, .wiggle_p img, .wiggle_n img", {
  force3D: true
});

gsap.to(".wiggle_a img", {
  y: -8,
  duration: 1.1,
  repeat: -1,
  yoyo: true,
  ease: "power1.inOut",
  force3D: true
});

gsap.to(".wiggle_p img", {
  y: -8,
  duration: 0.85,
  repeat: -1,
  yoyo: true,
  ease: "power1.inOut",
  force3D: true
});

gsap.to(".wiggle_n img", {
  y: -8,
  duration: 1,
  repeat: -1,
  yoyo: true,
  ease: "power1.inOut",
  force3D: true
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
      start: "10%-top center", // when the top of ___ hits the center of viewport
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
      start: "9.5%+top top", // when the top of ___ hits the center of viewport
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
      start: "25%+top top", // when the top of ___ hits the center of viewport
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
      start: "40%+top top", // when the top of ___ hits the center of viewport
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
      start: "55%+top top", // when the top of ___ hits the center of viewport
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
        trigger: ".act3_text",
        start: "bottom 80%+bottom",
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
    end: "+=1600",
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
  delay: 2.5 // x-second delay before this animation starts
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