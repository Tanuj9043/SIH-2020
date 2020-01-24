// This file is not being used

const socket    = io('http://localhost:8080');

// $('form').submit(function(){
//     socket.emit('realtime emotion', $('#m').val());
//     $('#m').val('');
//     return false;
// });

socket.on('emoHtml', msg => {
    msg = msg.split(";")
    pause = msg[4]
    $('#content').html(`<h2>${msg[0]}</h2><h4>${msg[1]}</h4>`);
    $('#content2').html(`<h2>${msg[2]}</h2><h4>${msg[3]}</h4>`);
    console.log(pause)
});