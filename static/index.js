function getBotResponse(){
    var myText = $("#my-text").val();
    var userBubble = '<div class="your-container"><div class="your-msg">'+ myText +'</div></div>';
    $("#my-text").val("");
    $(".chat-view").append(userBubble);
    $(".chat-view").stop().animate({scrollTop: $(".chat-view")[0].scrollHeight}, 1000);

    $.get("/get", {msg: myText }).done(function(data){
        var botBubble = '<div class="bot-container"><div class="bot-msg">'+ data +'</div></div>';
        $(".chat-view").append(botBubble);
    });
}
$("#my-text").keypress(function(e){
    if (e.which == 13){
        getBotResponse();
    }
});