const express   = require('express')
const app       = express()
const http      = require('http').Server(app);
const bodyParser = require('body-parser');
const ejs       = require('ejs');
const io        = require('socket.io')(8080);
const multer    = require('multer')
const path      = require('path')

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended : true }));
app.set('view engine', 'ejs');
app.use(express.static('public_static'))

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, './public_static/uploads')
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + '-' + file.originalname)
    }
  })
const upload    = multer({ storage : storage })

app.get('/', (req,res) => {
    res.render('index.ejs')
})

app.get('/video', (req,res)=>{
    url = req.query.url
    url = url.replace('watch?v=', 'embed/')
    url = url + '?enablejsapi=1'
    res.render('video.ejs', {url : url})
})

app.post('/video', (req,res)=>{
    redirectLink = '/video?url=' + req.body.videoLink
    res.redirect(redirectLink)
})

app.get('/localVideo', (req,res) => {
    res.render('localVideo.ejs', {url : req.query.url})
})

app.post('/localVideo', upload.single('local'), (req,res)=>{
    redirectLink = '/localVideo?url=' + req.file.filename
    res.redirect(redirectLink)
})

app.get('/mood', (req,res) => {
    res.render('mood.ejs')
})

app.get('/analysis', (req,res) => {
    res.render('analysis.ejs')
})

app.get('*', (req,res) => {
    res.redirect('/')
})

io.on('connection', socket => {
    socket.on('emoNode', msg => {
        io.emit('emoHtml', msg);
    });
  });
  
http.listen(3000, () => {
    console.log('Application started at https://localhost:3000');
})