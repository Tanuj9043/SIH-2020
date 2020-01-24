const socket    = io('http://localhost:8080');

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
    else {
        val = msg[5]%10
        $('#framediv').html(`<img src="/frames/fig${val}.png?${msg[5]}" height="450">`);
        // ? value keeps the url unique, while accessing the file with link before ?, this helps to beat cached images with old url
        $('#content').html(`<h2>${msg[0]}</h2><h4>${msg[1]}</h4>`);
        $('#content2').html(`<h2>${msg[2]}</h2><h4>${msg[3]}</h4>`);
    }

});