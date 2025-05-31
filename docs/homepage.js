// importing ScrollTrigger from GSAP
gsap.registerPlugin(ScrollTrigger);

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

lenis.on('scroll', ScrollTrigger.update)

// button scroll to scene1
document.querySelector('#dive_button').addEventListener('click', function () {
    const scene1 = document.querySelector('.scene1');

    if (scene1) {
        // calculate the scroll position
        const scene1Top = scene1.getBoundingClientRect().top + window.scrollY;
        const scrollTarget = scene1Top + (scene1.offsetHeight * 0.90);

        // scroll to the calculated position
        window.scrollTo({
            top: scrollTarget,
            behavior: 'smooth'
        });
    }
});

// Play or pause audio based on visibility of water_animation_container
document.addEventListener('DOMContentLoaded', () => {
  const audio = document.getElementById('bubbly-sound');
  const waterAnimationContainer = document.querySelector('.water_animation_container');

  // Adjust initial volume
  audio.volume = 0.3; // Set initial volume (0.0 to 1.0)

  // Use IntersectionObserver to monitor visibility
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        audio.play(); // Play audio when the container is visible
      } else {
        audio.pause(); // Pause audio when the container is not visible
      }
    });
  });

  observer.observe(waterAnimationContainer);

  // Fade out audio volume on scroll
  window.addEventListener('scroll', () => {
    const containerRect = waterAnimationContainer.getBoundingClientRect();
    const fadeStart = 0; // Start fading when the container is fully visible
    const fadeEnd = window.innerHeight; // End fading when the container is out of view

    // Calculate the fade factor based on scroll position
    const fadeFactor = Math.max(0, Math.min(1, containerRect.bottom / fadeEnd));

    // Adjust the audio volume based on the fade factor
    audio.volume = fadeFactor * 0.3; // Multiply by max volume (e.g., 0.5)
  });
});

// document.querySelector('#dive_button').addEventListener('click', function () {
//   window.location.href = '/projects'; 
// });

// button scroll to explore
document.querySelector('#explore_button').addEventListener('click', function () {
  window.location.href = '/projects'; 
});

// button scroll down by one viewport height when Dive In button is clicked
document.querySelector('#dive_button').addEventListener('click', function () {
  const targetScroll = window.scrollY + window.innerHeight; // Calculate the target scroll position
  const scrollStep = 10; // Adjust this value to control the speed (smaller = slower)
  const interval = 10; // Time between each scroll step in milliseconds

  const scrollInterval = setInterval(() => {
      const currentScroll = window.scrollY;
      if (currentScroll < targetScroll) {
          window.scrollBy(0, scrollStep); // Scroll by small steps
      } else {
          clearInterval(scrollInterval); // Stop scrolling when the target is reached
      }
  }, interval);
});

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
    start: "top 40%-top", // when the top of ___ hits the center of viewport
    end: "+=450", // adjust 
    scrub: true,
    markers: false
  }
});

tl3.to("#hand_container", {
  x: "100%"
})


gsap.fromTo("#project_container_1",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
      trigger: "#project_container_1",
      start: "top center", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

gsap.fromTo("#project_container_2",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
      trigger: "#project_container_2",
      start: "top center", // when the top of ___ hits the center of viewport
      toggleActions: "play none none reverse", // optional: makes it fade out on scroll up
      markers: false  
    }
  }
);

gsap.fromTo("#project_container_3",
  { opacity: 0, y: 20 },
  {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: "power2.out",
    scrollTrigger: {
      trigger: "#project_container_3",
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

window.addEventListener('load', () => {
  ScrollTrigger.refresh()
})

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