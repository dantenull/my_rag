import { createRouter, createWebHashHistory  } from 'vue-router'
import Chat from './components/Chat.vue'
import DocsManager from './components/DocsManager.vue'
import DocsQuestion from './components/DocsQuestion.vue'

const routes = [
    // { 
    //     path: '/', 
    //     component: App,
    //     children: [
    //         { path: '/chat', component: Chat, name: 'chat'},
    //     ]
    // },
    { path: '/chat', component: Chat, name: 'chat'},
    { path: '/docs/manage', component: DocsManager, name: 'docs_manage'},
    { path: '/docs/question', component: DocsQuestion, name: 'docs_question'},
]

const router = createRouter({
    history: createWebHashHistory(),
    routes, 
})

export default router;