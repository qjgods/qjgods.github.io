require([],function(){var i,e=!1,t=function(){require([yiliaConfig.rootUrl+"js/mobile.js"],function(i){i.init(),e=!0})},a=!1,n=function(){require([yiliaConfig.rootUrl+"js/pc.js"],function(i){i.init(),a=!0})},o={trident:(i=window.navigator.userAgent).indexOf("Trident")>-1,presto:i.indexOf("Presto")>-1,webKit:i.indexOf("AppleWebKit")>-1,gecko:i.indexOf("Gecko")>-1&&-1==i.indexOf("KHTML"),mobile:!!i.match(/AppleWebKit.*Mobile.*/),ios:!!i.match(/\(i[^;]+;( U;)? CPU.+Mac OS X/),android:i.indexOf("Android")>-1||i.indexOf("Linux")>-1,iPhone:i.indexOf("iPhone")>-1||i.indexOf("Mac")>-1,iPad:i.indexOf("iPad")>-1,webApp:-1==i.indexOf("Safari"),weixin:-1==i.indexOf("MicroMessenger")};$(window).bind("resize",function(){e&&a?$(window).unbind("resize"):$(window).width()>=700?n():t()}),o.mobile||$(window).width()<800?t():n(),resetTags=function(){for(var i=$(".tagcloud a"),e=0;e<i.length;e++){var t=Math.floor(7*Math.random());i.eq(e).addClass("color"+t)}$(".article-category a:nth-child(-n+2)").attr("class","color0")},yiliaConfig.fancybox&&require([yiliaConfig.fancybox_js],function(i){if(0!=$(".isFancy").length){for(var e=$(".article-inner img"),t=0,a=e.length;t<a;t++){var n=e.eq(t).attr("src");if(void 0===(o=e.eq(t).attr("alt")))var o=e.eq(t).attr("title");var c=e.eq(t).attr("width"),r=e.eq(t).attr("height");e.eq(t).replaceWith("<a href='"+n+"' title='"+o+"' rel='fancy-group' class='fancy-ctn fancybox'><img src='"+n+"' width="+c+" height="+r+" title='"+o+"' alt='"+o+"'></a>")}$(".article-inner .fancy-ctn").fancybox({type:"image"})}}),yiliaConfig.animate&&(yiliaConfig.isHome?require([yiliaConfig.scrollreveal],function(i){var e=["pulse","fadeIn","fadeInRight","flipInX","lightSpeedIn","rotateInUpLeft","slideInUp","zoomIn"],t=e.length,a=e[Math.ceil(Math.random()*t)-1];if(window.requestAnimationFrame){var n=".body-wrap > article",o=$(".body-wrap > article:first-child");if(o.height()>$(window).height()){n=".body-wrap > article:not(:first-child)";o.css({opacity:1})}i({duration:0,afterReveal:function(i){$(i).addClass("animated "+a).css({opacity:1})}}).reveal(n)}else if($(".body-wrap > article").css({opacity:1}),navigator.userAgent.match(/Safari/i)){function c(){$(".article").each(function(){$(this).offset().top<=$(window).scrollTop()+$(window).height()&&!$(this).hasClass("show")?($(this).removeClass("hidden").addClass("show"),$(this).addClass("is-hiddened")):$(this).hasClass("is-hiddened")||$(this).addClass("hidden")})}$(window).on("scroll",function(){c()}),c()}}):$(".body-wrap > article").css({opacity:1})),yiliaConfig.toc&&require(["toc"],function(){});var c=["#6da336","#ff945c","#66CC66","#99CC99","#CC6666","#76becc","#c99979","#918597","#4d4d4d"],r=Math.ceil(Math.random()*(c.length-1));$("#container .left-col .overlay").css({"background-color":c[r],opacity:.3}),$("#container #mobile-nav .overlay").css({"background-color":c[r],opacity:.7}),$("table").wrap("<div class='table-area'></div>"),$(document).ready(function(){$("#comments").length<1&&$("#scroll > a:nth-child(2)").hide()}),(yiliaConfig.isArchive||yiliaConfig.isTag||yiliaConfig.isCategory)&&$(document).ready(function(){$("#footer").after("<button class='hide-labels'>TAGS</button>"),$(".hide-labels").click(function(){$(".article-info").toggle(200)})}),$("ul > li").each(function(){var i={field:this.textContent.substring(0,2),check:function(i){var e=new RegExp(i);return this.field.match(e)}},e=["[ ]",["[x]","checked"]],t=i.check(e[1][0]),a=i.check(e[0]),n=$(this);function o(i,e){n.html(n.html().replace(i,"<input type='checkbox' "+e+"  >"))}(t||a)&&(this.classList.add("task-list"),t?(o(e[1][0],e[1][1]),this.classList.add("check")):o(e[0],""))})});