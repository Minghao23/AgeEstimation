# -*- coding: utf-8 -*-
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, ConversationHandler)
import src.my_request as my_req
import logging
import os


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

PHOTO = range(1)


def start(bot, update):
    user = update.message.from_user
    logger.info("Start with %s", user.first_name)
    update.message.reply_text(
        "你好！我是一个傻屌机器人，能猜出你的性别和年龄，请发给我一张你的自拍好么？你还可以和我对话，比如尝试说'你好'。如果不想再搭理我请输入 /cancel")

    return PHOTO


def echo(bot, update):
    user = update.message.from_user
    msg = update.message.text
    logger.info("Echo of %s: %s", user.first_name, msg)
    if msg == '你好':
        reply = '你好，我是你爸爸！'
    elif msg in ['再见', '拜拜', 'bye', '8', '88', '白白']:
        reply = '拜拜！'
        update.message.reply_text(reply)
        logger.info("User %s canceled the conversation.", user.first_name)
        return ConversationHandler.END
    else:
        reply = msg.replace("吗", "")
        reply = reply.replace("?", "!")
        reply = reply.replace("？", "!")
    update.message.reply_text(reply)
    update.message.reply_text("给我发一张照片吧！")

    return PHOTO


def photo(bot, update):
    user = update.message.from_user
    photo_file = bot.get_file(update.message.photo[-1].file_id)
    photo_path = '%s/images/user_photo.jpg' % root_dir
    photo_file.download(photo_path)
    logger.info("Photo of %s: %s", user.first_name, photo_path)
    update.message.reply_text('收到了，那我猜你的性别和年龄应该是……')

    data = {'photo_path': photo_path}
    response = my_req.detect(data)

    # handle exception
    if not response['successful']:
        if response['code'] == 2:
            update.message.reply_text('我想了很久也没猜到，大概是出问题了，我去报告一下，拜拜！')
            logger.info("Error of %s, code=%d", user.first_name, response['code'])
            return ConversationHandler.END
        if response['code'] == 1 or response['code'] == 3:
            update.message.reply_text('我脑子好像瓦特了，我去报告一下，拜拜！')
            logger.info("Error of %s, code=%d", user.first_name, response['code'])
            return ConversationHandler.END

    # output result
    results = response['results']
    result_photo_path = '%s/images/result_user_photo.jpg' % root_dir
    gender_map = {'F': '美女', 'M': '帅哥'}
    if len(results) == 0:
        update.message.reply_text('不好意思，我并没有在照片里看到你啊！')
    elif len(results) == 1:
        bot.send_photo(chat_id=update.message.chat_id, photo=open(result_photo_path, 'rb'))
        update.message.reply_text('你是一个%d岁的%s！' % (results[0]['age'], gender_map[results[0]['gender']]))
    else:
        bot.send_photo(chat_id=update.message.chat_id, photo=open(result_photo_path, 'rb'))
        update.message.reply_text('我在这张照片里看到了%d个人！分别是：' % len(results))
        for result in results:
            update.message.reply_text('%d岁的%s！' % (result['age'], gender_map[result['gender']]))
    update.message.reply_text('如果想再测一张，请再发送一张图片，如果不测了请输入 /cancel')

    return PHOTO


def cancel(bot, update):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('拜拜~')

    return ConversationHandler.END


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    token = "775029283:AAHNweyYIs9Zjp27HM3vlJTrK487fPr_fhU"
    request_args = {'proxy_url': 'http://127.0.0.1:1087'}
    updater = Updater(token, request_kwargs=request_args)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={

            PHOTO: [MessageHandler(Filters.photo, photo),
                    MessageHandler(Filters.text, echo)],

        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
