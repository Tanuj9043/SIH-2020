var myVideo = document.getElementById("video"); 
const socket    = io('http://localhost:8080');

cy = 0 //consecutive yes-pause commands
cn = 0 //consecutive no-pause commands

socket.on('emoHtml', msg => {
    msg = msg.split(";")
    if(msg[0]=="end"){
        $('#outerdiv').html(`
        <h2 class="text-light m-4">
            <b>Session ended.</b>
        </h2>
        <a href="/analysis" class="text-light"><button type="button" class="btn btn-lg btn-success">Show Analysis</button></a>
        `);
    }
    else if(msg[0]=="saving"){
        $('#outerdiv').html(`
        <h2 class="text-light m-4">
            <b>Session ended.</b>
        </h2>
        <img src="/loading.gif" height="30px" >
        <h2 style="display: inline" class="text-light mx-4">Preparing Analytics    </h2>
        <img src="/loading.gif" height="30px" >
        `);
    }
    else if(msg[0]=="pause"){ //session pause, not video pause
        $('#content').html(`<h4>Session Paused</h4>`);
        $('#content2').html(`<h4>Session Paused</h4>`);
    }
    else{
        pause = msg[4]
        $('#content').html(`<h2>${msg[0]}</h2><h4>${msg[1]}</h4>`);
        $('#content2').html(`<h2>${msg[2]}</h2><h4>${msg[3]}</h4>`);
        if(pause == "y"){
            cy += 1
            cn = 0
        }
        else{
            cy = 0
            cn += 1
        }

        if(cy>3){
            myVideo.pause();
            $('#playpause').html(`
            <span class="px-4 small">Playing</span>
            <b class="px-4 text-light">Paused</b>
            `);
        }
        
        if(cn>1){
            myVideo.play();
            $('#playpause').html(`
            <b class="px-4 text-light">Playing</b>
            <span class="px-4 small">Paused</span>
            `);
        }
    }
});
